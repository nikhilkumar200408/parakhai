import { useState, useCallback, useRef } from "react";

const API = "http://localhost:8000";

// ── Design tokens ─────────────────────────────────────────────────────
const C = {
  bg:       "#f0f2f8",
  surface:  "#ffffff",
  border:   "#dde2ee",
  text:     "#0d1b2a",
  sub:      "#4a5568",
  muted:    "#8896ab",
  accent:   "#1b4fff",
  accentBg: "#e8eeff",
  danger:   "#e63946",
  dangerBg: "#fff0f0",
  warn:     "#f4a261",
  warnBg:   "#fff7ed",
  ok:       "#2a9d4c",
  okBg:     "#edfaf3",
  mono:     "'JetBrains Mono','Fira Code',monospace",
  sans:     "'Inter','Segoe UI',system-ui,sans-serif",
};

const RISK_C = {
  HIGH:     { bg:C.dangerBg, text:C.danger,  border:"#f5c6cb" },
  MODERATE: { bg:C.warnBg,   text:C.warn,    border:"#ffd6a5" },
  LOW:      { bg:C.okBg,     text:C.ok,      border:"#b7e4c7" },
};
const SEV_C = {
  Severe:   C.danger,
  Moderate: C.warn,
  Mild:     C.ok,
};

// ── Utility components ────────────────────────────────────────────────
const Pill = ({ children, color, bg, border }) => (
  <span style={{
    display:"inline-block", padding:"3px 11px", borderRadius:20,
    fontSize:11, fontWeight:700, letterSpacing:0.8, fontFamily:C.sans,
    background:bg, color, border:`1px solid ${border}`,
  }}>{children}</span>
);

const Label = ({ children, style={} }) => (
  <div style={{
    fontSize:10, fontWeight:700, letterSpacing:1.5,
    textTransform:"uppercase", color:C.muted,
    fontFamily:C.sans, marginBottom:8, ...style,
  }}>{children}</div>
);

const Card = ({ children, style={} }) => (
  <div style={{
    background:C.surface, borderRadius:14,
    border:`1px solid ${C.border}`, padding:22, ...style,
  }}>{children}</div>
);

function Spinner() {
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:10, color:C.sub, fontFamily:C.sans,
    }}>
      <div style={{
        width:18, height:18, borderRadius:"50%",
        border:`2.5px solid ${C.border}`, borderTopColor:C.accent,
        animation:"spin 0.8s linear infinite",
      }}/>
      <span style={{ fontSize:13 }}>Running PARAKH analysis...</span>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

function NumberField({ label, value, onChange, min, max, step, hint }) {
  return (
    <div style={{ marginBottom:14 }}>
      <label style={{
        display:"block", fontSize:11, fontWeight:600, color:C.sub,
        fontFamily:C.sans, marginBottom:5,
      }}>{label}</label>
      <input type="number" min={min} max={max} step={step}
        value={value} onChange={e=>onChange(Number(e.target.value))}
        style={{
          width:"100%", padding:"8px 11px", borderRadius:8,
          border:`1px solid ${C.border}`, background:C.surface,
          fontFamily:C.mono, fontSize:13, color:C.text,
          outline:"none", boxSizing:"border-box",
          transition:"border-color .15s",
        }}
        onFocus={e=>e.target.style.borderColor=C.accent}
        onBlur={e=>e.target.style.borderColor=C.border}
      />
      {hint && <div style={{fontSize:10,color:C.muted,fontFamily:C.sans,marginTop:3}}>{hint}</div>}
    </div>
  );
}

function DropZone({ onFile, isDragging, setIsDragging }) {
  const ref = useRef();
  return (
    <div
      onClick={()=>ref.current?.click()}
      onDrop={e=>{
        e.preventDefault(); setIsDragging(false);
        const f=e.dataTransfer.files[0]; if(f) onFile(f);
      }}
      onDragOver={e=>{e.preventDefault();setIsDragging(true);}}
      onDragLeave={()=>setIsDragging(false)}
      style={{
        border:`2px dashed ${isDragging?C.accent:C.border}`,
        borderRadius:12, padding:"32px 20px", textAlign:"center",
        background:isDragging?C.accentBg:C.bg, cursor:"pointer",
        transition:"all .2s",
      }}
    >
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
           stroke={C.accent} strokeWidth="1.8" strokeLinecap="round"
           style={{margin:"0 auto 10px",display:"block"}}>
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      <div style={{fontSize:13,fontWeight:600,color:C.text,fontFamily:C.sans,marginBottom:3}}>
        Drop CT scan or chest X-ray here
      </div>
      <div style={{fontSize:11,color:C.muted,fontFamily:C.sans}}>
        PNG or JPEG — auto-detected modality
      </div>
      <input ref={ref} type="file" accept="image/png,image/jpeg"
             onChange={e=>e.target.files[0]&&onFile(e.target.files[0])}
             style={{display:"none"}}/>
    </div>
  );
}

function ZoneBar({ name, value }) {
  const pct = Math.round(value * 100);
  const col = value>=0.50 ? C.danger : value>=0.30 ? C.warn : C.ok;
  return (
    <div style={{ marginBottom:9 }}>
      <div style={{
        display:"flex", justifyContent:"space-between",
        marginBottom:4,
      }}>
        <span style={{fontSize:11,color:C.sub,fontFamily:C.sans}}>{name}</span>
        <span style={{fontSize:11,fontFamily:C.mono,color:col,fontWeight:700}}>
          {pct}%
        </span>
      </div>
      <div style={{
        height:8, borderRadius:4, background:C.border, overflow:"hidden",
      }}>
        <div style={{
          height:"100%", width:`${pct}%`, background:col,
          borderRadius:4, transition:"width .5s ease",
        }}/>
      </div>
    </div>
  );
}

function SummarySection({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const parsed = [];
  let current = null;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const isHeader = /^[A-Z][A-Z\s]+$/.test(trimmed) && trimmed.length < 40;
    if (isHeader) {
      if (current) parsed.push(current);
      current = { heading: trimmed, body: [] };
    } else if (current) {
      current.body.push(trimmed);
    }
  }
  if (current) parsed.push(current);

  return (
    <div style={{ fontFamily:C.sans }}>
      {parsed.map((sec,i) => (
        <div key={i} style={{ marginBottom:16 }}>
          <div style={{
            fontSize:10, fontWeight:700, letterSpacing:1.4,
            color:C.accent, marginBottom:5, textTransform:"uppercase",
          }}>{sec.heading}</div>
          <div style={{
            fontSize:13, color:C.text, lineHeight:1.75,
          }}>
            {sec.body.map((line,j) =>
              line.startsWith("-") ? (
                <div key={j} style={{
                  display:"flex", gap:8, marginBottom:3,
                }}>
                  <span style={{color:C.accent,fontWeight:700}}>—</span>
                  <span>{line.slice(1).trim()}</span>
                </div>
              ) : <div key={j}>{line}</div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────
export default function App() {
  const [file,       setFile]       = useState(null);
  const [preview,    setPreview]    = useState(null);
  const [fev,        setFev]        = useState(0.75);
  const [pack,       setPack]       = useState(10);
  const [age,        setAge]        = useState(55);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);
  const [dragging,   setDragging]   = useState(false);

  const handleFile = useCallback(f => {
    setFile(f); setResult(null); setError(null);
    const r = new FileReader();
    r.onload = e => setPreview(e.target.result);
    r.readAsDataURL(f);
  }, []);

  const analyse = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("fev1_fvc",   fev);
    fd.append("pack_years", pack);
    fd.append("age",        age);
    try {
      const res = await fetch(`${API}/analyze`, { method:"POST", body:fd });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail||"Analysis failed");
      }
      setResult(await res.json());
    } catch(e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const rc = result ? RISK_C[result.risk_level]||RISK_C.LOW : null;

  return (
    <div style={{ minHeight:"100vh", background:C.bg, fontFamily:C.sans }}>

      {/* Header */}
      <div style={{
        background:C.surface, borderBottom:`1px solid ${C.border}`,
        padding:"0 28px", height:56, display:"flex",
        alignItems:"center", justifyContent:"space-between",
        position:"sticky", top:0, zIndex:100,
        boxShadow:"0 1px 3px rgba(0,0,0,0.06)",
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:11 }}>
          <div style={{
            width:34, height:34, borderRadius:9,
            background:"linear-gradient(135deg,#1b4fff 0%,#00c2ff 100%)",
            display:"flex", alignItems:"center", justifyContent:"center",
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="white" strokeWidth="2.5" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <div>
            <div style={{fontSize:15,fontWeight:800,color:C.text,letterSpacing:0.3}}>
              PARAKH v2
            </div>
            <div style={{fontSize:9,color:C.muted,letterSpacing:1.2,textTransform:"uppercase"}}>
              Multimodal COPD Diagnostic System
            </div>
          </div>
        </div>
      </div>

      {/* Layout */}
      <div style={{
        maxWidth:1380, margin:"0 auto", padding:"24px 24px 48px",
        display:"grid", gridTemplateColumns:"320px 1fr", gap:20,
      }}>

        {/* Left column — inputs */}
        <div style={{ display:"flex", flexDirection:"column", gap:16 }}>

          <Card>
            <Label>Scan Upload</Label>
            <DropZone onFile={handleFile}
                      isDragging={dragging} setIsDragging={setDragging}/>
            {preview && (
              <div style={{
                marginTop:12, borderRadius:10, overflow:"hidden",
                border:`1px solid ${C.border}`,
              }}>
                <img src={preview} alt="scan"
                     style={{width:"100%",display:"block",
                             maxHeight:190,objectFit:"contain",background:"#000"}}/>
              </div>
            )}
          </Card>

          <Card>
            <Label>Clinical Parameters</Label>
            <NumberField label="FEV1/FVC Ratio" min={0.2} max={1.0} step={0.01}
              value={fev} onChange={setFev}
              hint="Normal >= 0.70   |   < 0.70 = obstructive"/>
            <NumberField label="Smoking History (pack-years)" min={0} max={120} step={0.5}
              value={pack} onChange={setPack}/>
            <NumberField label="Patient Age (years)" min={18} max={100} step={1}
              value={age} onChange={setAge}/>
          </Card>

          <button onClick={analyse} disabled={!file||loading} style={{
            width:"100%", padding:"13px 0", borderRadius:10,
            background: (!file||loading) ? C.border : C.accent,
            color: (!file||loading) ? C.muted : "#fff",
            border:"none", cursor: (!file||loading)?"not-allowed":"pointer",
            fontSize:14, fontWeight:700, fontFamily:C.sans,
            letterSpacing:0.3, transition:"background .2s",
          }}>
            {loading ? "Analysing..." : "Run PARAKH Analysis"}
          </button>

          {loading && <Spinner/>}

          {error && (
            <div style={{
              padding:"12px 14px", borderRadius:8,
              background:C.dangerBg, border:`1px solid #f5c6cb`,
              fontSize:12, color:C.danger, fontFamily:C.sans,
            }}>{error}</div>
          )}

          {result && (
            <Card style={{ borderTop:`3px solid ${rc.text}` }}>
              <Label>Quick Stats</Label>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
                {[
                  ["Probability", `${(result.probability*100).toFixed(1)}%`],
                  ["Severity",    result.severity],
                  ["Extent",      `${result.extent_pct?.toFixed(1)}%`],
                  ["Modality",    result.modality],
                  ["Anatomy",     result.detected_anatomy || "Lung"],
                ].map(([k,v])=>(
                  <div key={k} style={{
                    padding:"10px 12px", borderRadius:8,
                    background:C.bg, border:`1px solid ${C.border}`,
                  }}>
                    <div style={{fontSize:9,color:C.muted,
                                 letterSpacing:1.2,textTransform:"uppercase",
                                 marginBottom:3}}>{k}</div>
                    <div style={{fontSize:16,fontWeight:800,
                                 fontFamily:C.mono,
                                 color:k==="Severity"
                                   ?SEV_C[v]||C.text
                                   :k==="Probability"
                                     ?rc.text:C.text}}>
                      {v}
                    </div>
                  </div>
                ))}
              </div>
              <div style={{marginTop:12,display:"flex",gap:8,flexWrap:"wrap"}}>
                <Pill {...RISK_C[result.risk_level]}>{result.risk_level} RISK</Pill>
                <Pill bg={SEV_C[result.severity]+"22"}
                      color={SEV_C[result.severity]||C.text}
                      border={SEV_C[result.severity]||C.border}>
                  {result.severity}
                </Pill>
              </div>
            </Card>
          )}
        </div>

        {/* Right column — results */}
        <div style={{ display:"flex", flexDirection:"column", gap:20 }}>

          {!result && !loading && (
            <Card style={{
              display:"flex", flexDirection:"column",
              alignItems:"center", justifyContent:"center",
              minHeight:440, color:C.muted, gap:14,
            }}>
              <svg width="52" height="52" viewBox="0 0 24 24" fill="none"
                   stroke={C.border} strokeWidth="1.3">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M9 9h6M9 12h6M9 15h4"/>
              </svg>
              <div style={{fontSize:14,fontFamily:C.sans}}>
                Upload a CT scan or chest X-ray to begin analysis
              </div>
            </Card>
          )}

          {result && (
            <>
              {result.analysis_note && (
                <Card style={{
                  background:"#fff8e1", border:`1px solid #ffe082`,
                  color:"#66512c", padding:16, marginBottom:10,
                }}>
                  <div style={{fontSize:12,fontWeight:700,marginBottom:6}}>
                    Image anatomy note
                  </div>
                  <div style={{fontSize:13,lineHeight:1.6}}>
                    {result.analysis_note}
                  </div>
                </Card>
              )}
              {/* 8-panel figure */}
              <Card style={{ padding:16 }}>
                <Label>8-Panel Clinical Analysis</Label>
                <img
                  src={`data:image/png;base64,${result.figure_base64}`}
                  alt="PARAKH analysis panels"
                  style={{
                    width:"100%", borderRadius:10,
                    border:`1px solid ${C.border}`,
                  }}
                />
              </Card>

              {/* Zone severity */}
              <Card>
                <Label>Lung Zone Severity Breakdown</Label>
                <div style={{
                  display:"grid", gridTemplateColumns:"1fr 1fr", gap:"4px 28px",
                }}>
                  {result.zone_scores && Object.entries(result.zone_scores).map(([k,v])=>(
                    <ZoneBar key={k} name={k} value={v}/>
                  ))}
                </div>
                <div style={{
                  marginTop:12, padding:"10px 14px", borderRadius:8,
                  background:C.bg, border:`1px solid ${C.border}`,
                  fontSize:12, color:C.sub, fontFamily:C.sans,
                  display:"flex", gap:20, flexWrap:"wrap",
                }}>
                  <span>
                    <strong style={{color:C.ok}}>Low</strong> &lt; 30%
                  </span>
                  <span>
                    <strong style={{color:C.warn}}>Moderate</strong> 30-50%
                  </span>
                  <span>
                    <strong style={{color:C.danger}}>High</strong> &gt; 50%
                  </span>
                </div>
              </Card>

              {/* Hotspot table */}
              <Card>
                <Label>
                  Emphysema Hotspot Detection
                  ({result.hotspots?.length||0} region{result.hotspots?.length!==1?"s":""})
                </Label>
                {result.hotspots?.length > 0 ? (
                  <table style={{width:"100%",borderCollapse:"collapse",fontFamily:C.sans}}>
                    <thead>
                      <tr style={{borderBottom:`2px solid ${C.border}`}}>
                        {["Region","Position","Size","Confidence","Level"].map(h=>(
                          <th key={h} style={{
                            padding:"6px 10px", fontSize:10, fontWeight:700,
                            textAlign:"left", color:C.muted, letterSpacing:0.5,
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.hotspots.map((h,i)=>{
                        const c = h.confidence;
                        const cc = c>=0.60?C.danger:c>=0.40?C.warn:C.ok;
                        const lvl = c>=0.60?"High":c>=0.40?"Moderate":"Low";
                        return (
                          <tr key={i} style={{
                            borderBottom:`1px solid ${C.border}`,
                            background:i%2===0?C.surface:C.bg,
                          }}>
                            <td style={{padding:"8px 10px",fontSize:13,
                                        fontWeight:700,color:cc}}>R{i+1}</td>
                            <td style={{padding:"8px 10px",fontSize:11,
                                        fontFamily:C.mono,color:C.sub}}>
                              ({(h.x*100).toFixed(1)}%, {(h.y*100).toFixed(1)}%)
                            </td>
                            <td style={{padding:"8px 10px",fontSize:11,
                                        fontFamily:C.mono,color:C.sub}}>
                              {(h.w*100).toFixed(1)}% x {(h.h*100).toFixed(1)}%
                            </td>
                            <td style={{padding:"8px 10px"}}>
                              <div style={{display:"flex",alignItems:"center",gap:7}}>
                                <div style={{
                                  height:6,width:64,background:C.border,
                                  borderRadius:3,overflow:"hidden",
                                }}>
                                  <div style={{
                                    height:"100%",width:`${c*100}%`,
                                    background:cc,borderRadius:3,
                                  }}/>
                                </div>
                                <span style={{
                                  fontSize:11,fontFamily:C.mono,
                                  color:cc,fontWeight:700,
                                }}>
                                  {(c*100).toFixed(0)}%
                                </span>
                              </div>
                            </td>
                            <td style={{padding:"8px 10px"}}>
                              <Pill bg={cc+"22"} color={cc} border={cc}>{lvl}</Pill>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <div style={{
                    padding:"16px", borderRadius:8,
                    background:C.okBg, border:`1px solid #b7e4c7`,
                    fontSize:13, color:C.ok, fontFamily:C.sans,
                  }}>
                    No significant emphysema hotspots detected.
                  </div>
                )}
              </Card>

              {/* PARAKH clinical summary */}
              <Card style={{ borderTop:`3px solid ${C.accent}` }}>
                <div style={{
                  display:"flex", alignItems:"center",
                  justifyContent:"space-between", marginBottom:16,
                }}>
                  <Label style={{marginBottom:0}}>PARAKH Clinical Summary</Label>
                  <span style={{
                    fontSize:9, color:C.accent, fontWeight:700, letterSpacing:0.8,
                    padding:"3px 10px", borderRadius:10, background:C.accentBg,
                    fontFamily:C.sans,
                  }}>
                    PARAKH Clinical Summary
                  </span>
                </div>
                <SummarySection text={result.doctor_summary}/>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
