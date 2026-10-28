"""
PARAKH v2 Clinical Dashboard - Backend API
Supports: Chest X-ray AND CT scan (axial PNG/JPEG slices)
New:      Lung zone severity map, multi-panel severity quantification,
          train-computed normalization loaded from parakh_metadata.json
"""

import os, io, base64, json, math, traceback
from typing import Optional, List

import numpy as np
import cv2
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import timm

# ── Config ────────────────────────────────────────────────────────────
WEIGHTS_PATH  = os.environ.get("PARAKH_WEIGHTS",  "./parakh_weights.pth")
META_PATH     = os.environ.get("PARAKH_META",     "./parakh_metadata.json")
OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL    = "https://api.openai.com/v1/chat/completions"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE      = 192
HR_SIZE       = 512

# ── Load normalization stats saved during training ────────────────────
def load_norm_stats():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        mean = np.array(meta["norm_mean"], dtype=np.float32)
        std  = np.array(meta["norm_std"],  dtype=np.float32)
        print(f"[PARAKH] Loaded norm stats from {META_PATH}  "
              f"mean={mean[0]:.4f}  std={std[0]:.4f}")
        return mean, std
    print("[PARAKH] metadata.json not found — using ImageNet defaults")
    return (np.array([0.485,0.456,0.406], dtype=np.float32),
            np.array([0.229,0.224,0.225], dtype=np.float32))

NORM_MEAN, NORM_STD = load_norm_stats()

# ── Model (same architecture as training) ────────────────────────────
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class TabularEncoder(nn.Module):
    def __init__(self, in_f=3, hidden=128, embed=1024, n_q=4, drop=0.2):
        super().__init__()
        self.n_q, self.embed = n_q, embed
        self.net = nn.Sequential(
            nn.Linear(in_f,   hidden), Swish(), nn.Dropout(drop),
            nn.Linear(hidden, hidden*2), Swish(), nn.Dropout(drop),
            nn.Linear(hidden*2, embed*n_q),
        )
    def forward(self, x): return self.net(x).view(-1, self.n_q, self.embed)

class SwinV2Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "swinv2_base_window12_192.ms_in22k",
            pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.feature_dim = self.backbone.num_features
    def forward(self, x):
        f = self.backbone.forward_features(x)
        if f.dim() == 4:
            b, h, w, c = f.shape; f = f.reshape(b, h*w, c)
        return f

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed=1024, heads=8, drop=0.1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(embed, heads, drop, batch_first=True)
        self.norm_q = nn.LayerNorm(embed); self.norm_kv = nn.LayerNorm(embed)
        self.norm_o = nn.LayerNorm(embed)
        self.ffn = nn.Sequential(
            nn.Linear(embed, embed*2), Swish(), nn.Dropout(drop),
            nn.Linear(embed*2, embed),
        )
    def forward(self, q, kv):
        a, w = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                         need_weights=True, average_attn_weights=True)
        x = q + a; return x + self.ffn(self.norm_o(x)), w

class ParakhSwinV2(nn.Module):
    def __init__(self, pretrained=False, n_q=4, heads=8, drop=0.2):
        super().__init__()
        self.visual  = SwinV2Backbone(pretrained)
        dim          = self.visual.feature_dim
        self.tabular = TabularEncoder(3, 128, dim, n_q, drop)
        self.fusion  = CrossAttentionFusion(dim, heads, drop)
        self.head    = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim//2),
            Swish(), nn.Dropout(drop), nn.Linear(dim//2, 1),
        )
        self.attn_weights = None
    def forward(self, img, ehr):
        v = self.visual(img); t = self.tabular(ehr)
        f, w = self.fusion(t, v); self.attn_weights = w
        return self.head(f.mean(1)).squeeze(-1)

_model: Optional[ParakhSwinV2] = None
def get_model():
    global _model
    if _model is not None: return _model
    m = ParakhSwinV2(pretrained=False).to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"[PARAKH] Weights loaded from {WEIGHTS_PATH}")
    else:
        print(f"[PARAKH] WARNING: No weights at {WEIGHTS_PATH} — DEMO mode")
    m.eval(); _model = m; return _model

# ── Image preprocessing (CT or X-ray) ────────────────────────────────
def detect_modality(img_gray: np.ndarray) -> str:
    """
    Heuristic modality detection.
    CT slices tend to have more extreme bright spots (bone/air) and
    a characteristic circular field-of-view (black corners).
    Returns 'CT' or 'XRAY'.
    """
    h, w    = img_gray.shape
    corners = [
        img_gray[:h//8, :w//8].mean(),
        img_gray[:h//8, -w//8:].mean(),
        img_gray[-h//8:, :w//8].mean(),
        img_gray[-h//8:, -w//8:].mean(),
    ]
    if np.mean(corners) < 12 and img_gray.max() > 200:
        return "CT"
    return "XRAY"


def detect_anatomy(img_gray: np.ndarray) -> str:
    h, w = img_gray.shape
    aspect = float(h) / max(w, 1)
    corners = [
        img_gray[:h//8, :w//8].mean(),
        img_gray[:h//8, -w//8:].mean(),
        img_gray[-h//8:, :w//8].mean(),
        img_gray[-h//8:, -w//8:].mean(),
    ]
    corner_dark = np.mean(corners) < 20 and img_gray.max() > 200
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(edges.mean() / 255.0)
    center = img_gray[h//4:3*h//4, w//4:3*w//4]
    mean_val = float(img_gray.mean())

    left = img_gray[:, :w//2]
    right = img_gray[:, w//2:]
    left_dark = float((left < 110).mean())
    right_dark = float((right < 110).mean())
    center_dark = float((center < 100).mean())

    # CT lung slice: circular FOV with dark lungs and bright bone/air contrast
    if corner_dark and 0.9 <= aspect <= 1.20:
        if left_dark > 0.18 and right_dark > 0.18 and edge_density > 0.06:
            return "Lung"
        if mean_val > 140:
            return "Abdomen/Pelvis"
        return "Spine"

    # Chest X-ray: wide image with bilateral lung fields
    if w > h and aspect < 1.5:
        if left_dark > 0.18 and right_dark > 0.18:
            return "Lung"
        if center.mean() > 140 and edge_density < 0.08:
            return "Extremity/Joint"
        if center.mean() > 100 and edge_density > 0.12:
            return "Head/Brain"

    if aspect >= 1.25:
        if mean_val < 110:
            return "Head/Brain"
        return "Abdomen/Pelvis"

    if aspect < 0.80:
        return "Extremity/Joint"

    if 0.85 <= aspect <= 1.15:
        if center_dark > 0.15 and edge_density > 0.07:
            return "Lung"
        return "Head/Brain"

    return "Unknown anatomy"


def preprocess(raw_bytes: bytes):
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot decode image. Upload PNG or JPEG.")

    modality = detect_modality(img)
    anatomy = detect_anatomy(img)
    original = img.copy()

    if modality == "CT":
        # CT: window/level (lung window: W=1500, L=-600)
        # Since we only have 8-bit here, simulate windowing via CLAHE
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        # Remove circular FOV black corners via thresholding
        _, mask = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
        enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    else:
        # Chest X-ray: standard CLAHE
        clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(img)

    hr      = cv2.resize(enhanced, (HR_SIZE, HR_SIZE), interpolation=cv2.INTER_CUBIC)
    hr_rgb  = cv2.cvtColor(hr, cv2.COLOR_GRAY2RGB)

    inp     = cv2.resize(hr_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    inp     = inp.astype(np.float32) / 255.0
    inp     = (inp - NORM_MEAN) / NORM_STD
    tensor  = torch.from_numpy(inp.transpose(2,0,1)).unsqueeze(0).to(DEVICE)

    return original, hr_rgb, tensor, modality, anatomy


def make_ehr(fev1_fvc, pack_years, age):
    arr = np.array([fev1_fvc, pack_years/100.0, age/100.0], dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

# ── Grad-CAM ──────────────────────────────────────────────────────────
class GradCAMHook:
    def __init__(self, model):
        self.grads = None; self.acts = None; self._h = []
        tgt = model.visual.backbone.norm
        self._h.append(tgt.register_forward_hook(
            lambda m,i,o: setattr(self,"acts",o.detach())))
        self._h.append(tgt.register_full_backward_hook(
            lambda m,gi,go: setattr(self,"grads",go[0].detach())))

    def compute(self, model, img_t, ehr_t):
        model.zero_grad()
        logit = model(img_t, ehr_t)
        logit.backward()
        g, a = self.grads, self.acts
        if g is None or a is None:
            return np.zeros((HR_SIZE, HR_SIZE), dtype=np.float32)
        if g.dim() == 4:
            b,h,w,c = g.shape; g = g.reshape(b,h*w,c); a = a.reshape(b,h*w,c)
        w_  = g.mean(1, keepdim=True)
        cam = F.relu((w_*a).sum(-1))
        n   = cam.shape[-1]; s = int(math.sqrt(n))
        cam = cam.reshape(1,1,s,s)
        cam = F.interpolate(cam, (HR_SIZE,HR_SIZE), mode="bilinear",
                            align_corners=False).squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam.astype(np.float32)

    def remove(self):
        for h in self._h: h.remove()


def run_gradcam(model, img_t, ehr_t):
    hook = GradCAMHook(model)
    model.train()
    cam = hook.compute(model, img_t, ehr_t)
    hook.remove(); model.eval()
    return cam

# ── Severity quantification ───────────────────────────────────────────
ZONES = {
    "Upper Right":  (0.00, 0.33, 0.50, 1.00),
    "Upper Left":   (0.00, 0.33, 0.00, 0.50),
    "Middle Right": (0.33, 0.66, 0.50, 1.00),
    "Middle Left":  (0.33, 0.66, 0.00, 0.50),
    "Lower Right":  (0.66, 1.00, 0.50, 1.00),
    "Lower Left":   (0.66, 1.00, 0.00, 0.50),
}

def quantify_severity(cam: np.ndarray, threshold: float = 0.40):
    """
    Returns:
      extent_pct    : % of lung area above threshold
      zone_scores   : per-zone activation mean
      hotspots      : list of bounding-box dicts
      severity_label: Mild / Moderate / Severe
    """
    h, w       = cam.shape
    binary     = (cam >= threshold).astype(np.uint8)
    extent_pct = 100.0 * binary.sum() / (h * w)

    zone_scores = {}
    for name, (r0, r1, c0, c1) in ZONES.items():
        roi = cam[int(r0*h):int(r1*h), int(c0*w):int(c1*w)]
        zone_scores[name] = float(roi.mean())

    contours, _ = cv2.findContours(
        (cam * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    _, binary2 = cv2.threshold(
        (cam*255).astype(np.uint8), int(threshold*255), 255, cv2.THRESH_BINARY
    )
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    hotspots = []
    for cnt in contours2:
        if cv2.contourArea(cnt) < 80: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        hotspots.append({
            "x": round(x/w, 4), "y": round(y/h, 4),
            "w": round(bw/w, 4), "h": round(bh/h, 4),
            "confidence": round(float(cam[y:y+bh, x:x+bw].mean()), 4),
        })
    hotspots.sort(key=lambda b: b["confidence"], reverse=True)
    hotspots = hotspots[:8]

    if extent_pct < 15:    sev = "Mild"
    elif extent_pct < 35:  sev = "Moderate"
    else:                  sev = "Severe"

    return extent_pct, zone_scores, hotspots, sev


# ── Dashboard figure (8 panels) ───────────────────────────────────────
EMPHYSEMA_CMAP = LinearSegmentedColormap.from_list(
    "emphysema", ["#0d1b2a","#1b4fff","#00c2ff","#f4a261","#e63946"]
)

def build_dashboard(original, hr_rgb, cam, prob,
                    extent_pct, zone_scores, hotspots, severity,
                    fev1_fvc, pack_years, age, modality):
    fig = plt.figure(figsize=(22, 14), facecolor="white")
    fig.suptitle(
        f"PARAKH v2  |  {modality} Scan Analysis",
        fontsize=15, fontweight="bold", color="#0d1b2a", y=0.99
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40,
                           wspace=0.32, left=0.04, right=0.98,
                           top=0.93, bottom=0.06)

    # Panel A — Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original, cmap="gray")
    ax.set_title(f"A  Original ({modality})", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.axis("off")

    # Panel B — CLAHE enhanced
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hr_rgb)
    ax.set_title("B  CLAHE Enhanced 512px", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.axis("off")

    # Panel C — Grad-CAM overlay with hotspot boxes
    ax = fig.add_subplot(gs[0, 2])
    base  = hr_rgb.astype(np.float32) / 255.0
    heat  = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat  = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    over  = np.clip(0.55*base + 0.45*heat, 0, 1)
    ax.imshow(over)
    H, W  = HR_SIZE, HR_SIZE
    for i, box in enumerate(hotspots):
        rect = patches.Rectangle(
            (box["x"]*W, box["y"]*H), box["w"]*W, box["h"]*H,
            linewidth=2, edgecolor="#e63946", facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(box["x"]*W+2, box["y"]*H-4,
                f"R{i+1} {box['confidence']*100:.0f}%",
                color="#e63946", fontsize=8, fontweight="bold")
    ax.set_title("C  Grad-CAM + Hotspots", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.axis("off")

    # Panel D — Severity heatmap (CAM as standalone colormap)
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(cam, cmap=EMPHYSEMA_CMAP, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Activation intensity")
    ax.set_title("D  Severity Heatmap", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.axis("off")

    # Panel E — Lung zone severity bar chart
    ax = fig.add_subplot(gs[1, 0])
    zones  = list(zone_scores.keys())
    scores = list(zone_scores.values())
    colors = ["#e63946" if s >= 0.5 else
              "#f4a261" if s >= 0.3 else "#2a9d4c" for s in scores]
    bars = ax.barh(zones, scores, color=colors, height=0.55,
                   edgecolor="white", linewidth=1.2)
    ax.axvline(x=0.40, color="#f4a261", linewidth=1.5,
               linestyle="--", alpha=0.9, label="Threshold 0.40")
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, scores):
        ax.text(val+0.02, bar.get_y()+bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Activation", fontsize=9, color="#5a6480")
    ax.set_title("E  Zone Severity", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#5a6480", labelsize=8)

    # Panel F — Risk gauge
    ax = fig.add_subplot(gs[1, 1])
    theta = np.linspace(0, np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e4e8f0", lw=20,
            solid_capstyle="round")
    end    = max(1, int(prob*300))
    gcol   = "#e63946" if prob>=0.65 else "#f4a261" if prob>=0.40 else "#2a9d4c"
    ax.plot(np.cos(theta[:end]), np.sin(theta[:end]),
            color=gcol, lw=20, solid_capstyle="round")
    ax.text(0, 0.28, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color=gcol)
    ax.text(0, 0.08, "Emphysema Risk", ha="center", fontsize=10,
            color="#5a6480")
    sev_c = {"Mild":"#2a9d4c","Moderate":"#f4a261","Severe":"#e63946"}
    ax.text(0, -0.08, severity, ha="center", fontsize=12,
            fontweight="bold", color=sev_c.get(severity,"#0d1b2a"))
    ax.text(-1.0,-0.12,"Low",  fontsize=9, color="#2a9d4c", fontweight="bold")
    ax.text( 0.82,-0.12,"High", fontsize=9, color="#e63946", fontweight="bold")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.20,1.15)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("F  Risk Gauge", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)

    # Panel G — Clinical features
    ax = fig.add_subplot(gs[1, 2])
    features = {
        "FEV1/FVC\n(ratio)":     (fev1_fvc,    0.70, 1.0),
        "Pack-years\n(norm)":    (pack_years/100, 0.0, 1.0),
        "Age\n(norm)":           (age/100,        0.0, 1.0),
        "Extent%\n(/100)":       (extent_pct/100, 0.0, 1.0),
    }
    names  = list(features.keys())
    vals   = [v[0] for v in features.values()]
    refs   = [v[1] for v in features.values()]
    fcols  = []
    for i, (v, r, _) in enumerate(features.values()):
        if i == 0: fcols.append("#e63946" if v < r else "#2a9d4c")
        elif i == 3: fcols.append("#e63946" if v > 0.35 else
                                  "#f4a261" if v > 0.15 else "#2a9d4c")
        else: fcols.append("#1b4fff")
    ax.barh(names, vals, color=fcols, height=0.5, edgecolor="white",
            linewidth=1.2)
    ax.axvline(x=0.70, color="#f4a261", linewidth=1.5, linestyle="--",
               alpha=0.9, label="FEV1/FVC ref")
    ax.set_xlim(0, 1.1)
    for i, (v, bar) in enumerate(zip(vals,
            ax.patches)):
        ax.text(v+0.02, bar.get_y()+bar.get_height()/2,
                f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Normalised Value", fontsize=9, color="#5a6480")
    ax.set_title("G  Clinical Features", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#5a6480", labelsize=8)

    # Panel H — Hotspot confidence chart
    ax = fig.add_subplot(gs[1, 3])
    if hotspots:
        hlabels = [f"R{i+1}" for i in range(len(hotspots))]
        hvals   = [h["confidence"] for h in hotspots]
        hcols   = ["#e63946" if c>=0.60 else
                   "#f4a261" if c>=0.40 else "#2a9d4c" for c in hvals]
        ax.bar(hlabels, hvals, color=hcols, edgecolor="white", linewidth=1.2)
        ax.axhline(y=0.40, color="#f4a261", linewidth=1.5, linestyle="--",
                   label="Detection threshold")
        ax.set_ylim(0, 1.1)
        for bar, v in zip(ax.patches, hvals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                    f"{v:.2f}", ha="center", fontsize=9)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No hotspots\ndetected", ha="center", va="center",
                fontsize=12, color="#5a6480", transform=ax.transAxes)
    ax.set_ylabel("Activation Confidence", fontsize=9, color="#5a6480")
    ax.set_title("H  Hotspot Confidence", fontweight="bold",
                  fontsize=11, color="#0d1b2a", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#5a6480")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── LLM doctor summary ────────────────────────────────────────────────
def openai_summary(prompt: str) -> str:
    if not OPENAI_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_KEY}",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 420,
    }
    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("choices"):
        return data["choices"][0].get("message", {}).get("content", "").strip()
    return json.dumps(data)


def doctor_summary(prob, fev1_fvc, pack_years, age,
                   hotspots, extent_pct, severity, modality, anatomy):
    if anatomy != "Lung":
        anatomy_note = (
            f"DETECTED ANATOMY: {anatomy}. "
            "This image appears outside the model's primary pulmonary domain."
        )
    else:
        anatomy_note = ""
    if not OPENAI_KEY:
        return _fallback(prob, fev1_fvc, pack_years, age,
                         hotspots, extent_pct, severity, modality, anatomy)
    n_spots = len(hotspots)
    risk    = "HIGH" if prob>=0.65 else "MODERATE" if prob>=0.40 else "LOW"
    prompt  = f"""You are a specialist radiologist AI assistant in the PARAKH
clinical decision support system. Generate a structured clinical summary
for a reviewing pulmonologist.

SCAN MODALITY: {modality}
DETECTED ANATOMY: {anatomy}
EMPHYSEMA PROBABILITY: {prob*100:.1f}% (Risk: {risk})
SEVERITY CLASSIFICATION: {severity}
LUNG FIELD INVOLVEMENT: {extent_pct:.1f}% of imaged area above activation threshold
HOTSPOT REGIONS DETECTED: {n_spots}
FEV1/FVC RATIO: {fev1_fvc:.2f} (normal >= 0.70)
SMOKING HISTORY: {pack_years:.0f} pack-years
PATIENT AGE: {age:.0f} years

{anatomy_note}

Write a structured report with these exact section headers on their own line:
FINDINGS
CLINICAL CORRELATION
SEVERITY ASSESSMENT
RECOMMENDED NEXT STEPS
DISCLAIMER

Rules: No emojis. Clinical register. Under 260 words total.
Bullet points only in RECOMMENDED NEXT STEPS (3 bullets)."""
    try:
        return openai_summary(prompt)
    except Exception as e:
        print(f"[PARAKH] OpenAI LLM failed: {e}")
        return _fallback(prob, fev1_fvc, pack_years, age,
                         hotspots, extent_pct, severity, modality, anatomy)


def _fallback(prob, fev1_fvc, pack_years, age,
              hotspots, extent_pct, severity, modality, anatomy):
    risk = "HIGH" if prob>=0.65 else "MODERATE" if prob>=0.40 else "LOW"
    fev_s = ("below normal (obstructive pattern)" if fev1_fvc < 0.70
             else "within normal limits")
    anatomy_note = ""
    if anatomy != "Lung":
        anatomy_note = f"\nNOTE: Detected anatomy appears to be {anatomy}; pulmonary analysis may be limited."
    return f"""FINDINGS
PARAKH SwinV2 multimodal analysis of the {modality} scan assigned an emphysema \
probability of {prob*100:.1f}%. Grad-CAM activation mapping identified \
{len(hotspots)} high-confidence region(s) covering approximately \
{extent_pct:.1f}% of the imaged lung field.{anatomy_note}

CLINICAL CORRELATION
FEV1/FVC ratio of {fev1_fvc:.2f} is {fev_s}. A {pack_years:.0f} \
pack-year smoking history in a {age:.0f}-year-old patient provides additional \
clinical context consistent with obstructive airway disease risk.

SEVERITY ASSESSMENT
Overall emphysema burden is classified as {severity} based on spatial \
activation extent. Risk level: {risk} ({prob*100:.1f}% model confidence).

RECOMMENDED NEXT STEPS
- Formal spirometry (pre- and post-bronchodilator FEV1, FVC, TLC)
- High-resolution CT thorax if not already performed for anatomical staging
- Pulmonology referral for GOLD stage assessment and management planning
"""


# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="PARAKH v2 Clinical API",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    get_model()
    print(f"[PARAKH v2] Ready on {DEVICE}")

@app.get("/health")
async def health():
    return {"status":"ok","device":str(DEVICE),"version":"2.0.0"}

@app.post("/analyze")
async def analyze(
    file:       UploadFile = File(...),
    fev1_fvc:   float = Form(0.75),
    pack_years: float = Form(10.0),
    age:        float = Form(55.0),
):
    if file.content_type not in ("image/png","image/jpeg","image/jpg"):
        raise HTTPException(400, "Upload PNG or JPEG only.")

    raw = await file.read()
    try:
        original, hr_rgb, img_t, modality, anatomy = preprocess(raw)
    except Exception as e:
        raise HTTPException(422, f"Preprocessing failed: {e}")

    ehr_t = make_ehr(fev1_fvc, pack_years, age)
    model = get_model()

    with torch.no_grad():
        prob = float(torch.sigmoid(model(img_t, ehr_t)).item())

    try:
        cam = run_gradcam(model, img_t, ehr_t)
    except Exception as e:
        print(f"[PARAKH] Grad-CAM error: {e}")
        cam = np.zeros((HR_SIZE, HR_SIZE), dtype=np.float32)

    extent_pct, zone_scores, hotspots, severity = quantify_severity(cam)

    fig_b64 = build_dashboard(
        original, hr_rgb, cam, prob,
        extent_pct, zone_scores, hotspots, severity,
        fev1_fvc, pack_years, age, modality
    )

    summary = doctor_summary(prob, fev1_fvc, pack_years, age,
                              hotspots, extent_pct, severity, modality, anatomy)

    risk = "HIGH" if prob>=0.65 else "MODERATE" if prob>=0.40 else "LOW"
    analysis_note = None
    if anatomy != "Lung":
        analysis_note = (
            f"Detected anatomy: {anatomy}. This image appears outside the model's primary pulmonary domain."
        )

    response = {
        "probability":    round(prob, 4),
        "risk_level":     risk,
        "severity":       severity,
        "extent_pct":     round(extent_pct, 2),
        "modality":       modality,
        "detected_anatomy": anatomy,
        "zone_scores":    {k: round(v,4) for k,v in zone_scores.items()},
        "hotspots":       hotspots,
        "figure_base64":  fig_b64,
        "doctor_summary": summary,
        "model_info":     {"name":"Parakh-SwinV2","version":"2.0","device":str(DEVICE)},
    }
    if analysis_note:
        response["analysis_note"] = analysis_note

    return JSONResponse(response) 