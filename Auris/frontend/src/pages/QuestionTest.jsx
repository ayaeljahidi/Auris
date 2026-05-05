import { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://localhost:8000";

const SAMPLE = `Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications. Its architecture is based on a Master-Slave model where the Control Plane makes all strategic decisions while Worker Nodes host the actual microservices. The Control Plane has four components: the API Server which handles authentication and validation, the Cluster Store based on etcd using the Raft algorithm for strong consistency, the Scheduler which places Pods through a two-phase filtering and ranking process, and the Controller Manager which permanently monitors the gap between desired and actual state. Self-healing is achieved automatically when the Controller Manager detects a crashed Pod and replaces it without human intervention. Rolling Updates allow deployments to update Pods one by one with zero downtime, and a single rollback command restores the previous version if a bug is detected.`;

/* ─── tiny keyframe injector ─────────────────────────────────────── */
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  @keyframes fadeUp   { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
  @keyframes spin     { to   { transform: rotate(360deg); } }
  @keyframes scanline { 0%,100% { opacity:.03; } 50% { opacity:.07; } }
  @keyframes blink    { 0%,100% { opacity:1; } 50% { opacity:0; } }
  @keyframes glow-pulse { 0%,100% { box-shadow: 0 0 20px rgba(56,189,248,.15); }
                          50%     { box-shadow: 0 0 42px rgba(56,189,248,.35); } }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body { background: #060912; }

  .qt-page {
    min-height: 100vh;
    background: #060912;
    font-family: 'Syne', sans-serif;
    color: #cdd6f4;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 56px 20px 100px;
    position: relative;
    overflow-x: hidden;
  }

  /* scanline texture */
  .qt-page::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(255,255,255,.013) 2px,
      rgba(255,255,255,.013) 4px
    );
    pointer-events: none; z-index: 0;
    animation: scanline 4s ease-in-out infinite;
  }

  /* top ambient glow */
  .qt-glow {
    position: fixed; top: -180px; left: 50%;
    transform: translateX(-50%);
    width: 900px; height: 500px;
    background: radial-gradient(ellipse at center,
      rgba(56,189,248,.08) 0%,
      rgba(99,102,241,.06) 40%,
      transparent 70%);
    pointer-events: none; z-index: 0;
  }

  .qt-content {
    position: relative; z-index: 1;
    width: 100%; max-width: 820px;
  }

  /* ── header ── */
  .qt-header {
    text-align: center; margin-bottom: 52px;
    animation: fadeUp .6s ease both;
  }

  .qt-brand {
    display: inline-flex; align-items: center;
    gap: 12px; margin-bottom: 18px;
  }

  .qt-brand-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 28px rgba(56,189,248,.35);
  }

  .qt-brand-name {
    font-size: 2rem; font-weight: 800; letter-spacing: -.04em;
    background: linear-gradient(135deg, #e2e8f0 30%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }

  .qt-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(56,189,248,.07);
    border: 1px solid rgba(56,189,248,.18);
    border-radius: 999px; padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; color: #38bdf8; letter-spacing: .06em;
  }

  .qt-pill-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
  }

  /* ── frame / panel ── */
  .qt-panel {
    background: rgba(14,18,32,.85);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 18px;
    backdrop-filter: blur(20px);
    animation: fadeUp .6s ease both;
  }

  .qt-panel-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 18px;
  }

  .qt-panel-label {
    font-size: .68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .14em;
    color: #38bdf8;
  }

  .qt-panel-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,.25), transparent);
  }

  /* the text input frame */
  .qt-textarea-wrap {
    position: relative;
    border: 1px solid rgba(56,189,248,.18);
    border-radius: 14px;
    overflow: hidden;
    transition: border-color .2s, box-shadow .2s;
  }

  .qt-textarea-wrap.focused {
    border-color: rgba(56,189,248,.5);
    box-shadow: 0 0 0 3px rgba(56,189,248,.08),
                inset 0 0 30px rgba(56,189,248,.03);
  }

  .qt-textarea-wrap textarea {
    width: 100%;
    min-height: 190px;
    background: rgba(6,9,18,.9);
    border: none; outline: none;
    color: #cdd6f4;
    font-family: 'JetBrains Mono', monospace;
    font-size: .88rem; line-height: 1.75;
    padding: 18px 20px;
    resize: vertical;
    caret-color: #38bdf8;
  }

  .qt-textarea-wrap textarea::placeholder { color: rgba(148,163,184,.35); }

  /* char counter */
  .qt-char-count {
    position: absolute; bottom: 10px; right: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; color: rgba(148,163,184,.4);
    pointer-events: none;
  }

  /* cursor blink in empty state */
  .qt-cursor {
    display: inline-block; width: 2px; height: .85em;
    background: #38bdf8; margin-left: 2px; vertical-align: middle;
    animation: blink 1s step-end infinite;
  }

  /* ── buttons ── */
  .qt-btn-row {
    display: flex; gap: 10px; margin-top: 16px;
  }

  .qt-btn-primary {
    flex: 1; padding: 14px;
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: #fff; border: none; border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: .95rem; font-weight: 700; letter-spacing: .01em;
    cursor: pointer;
    box-shadow: 0 4px 24px rgba(56,189,248,.25);
    transition: opacity .2s, transform .1s, box-shadow .2s;
    animation: glow-pulse 3s ease-in-out infinite;
  }

  .qt-btn-primary:hover:not(:disabled) {
    opacity: .92; transform: translateY(-1px);
    box-shadow: 0 8px 32px rgba(56,189,248,.4);
  }

  .qt-btn-primary:active:not(:disabled) { transform: translateY(0); }

  .qt-btn-primary:disabled {
    background: rgba(56,189,248,.12);
    color: rgba(148,163,184,.4); cursor: not-allowed;
    box-shadow: none; animation: none;
  }

  .qt-btn-ghost {
    padding: 14px 18px;
    background: transparent;
    color: rgba(148,163,184,.7);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: .85rem; cursor: pointer;
    transition: border-color .2s, color .2s;
    white-space: nowrap;
  }

  .qt-btn-ghost:hover {
    border-color: rgba(56,189,248,.3); color: #38bdf8;
  }

  /* ── status bar ── */
  .qt-status {
    display: flex; align-items: center; gap: 12px;
    margin-top: 16px;
    background: rgba(56,189,248,.05);
    border: 1px solid rgba(56,189,248,.12);
    border-radius: 10px; padding: 13px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem; color: #38bdf8;
  }

  .qt-spinner {
    width: 16px; height: 16px; flex-shrink: 0;
    border: 2px solid rgba(56,189,248,.15);
    border-top-color: #38bdf8;
    border-radius: 50%;
    animation: spin .75s linear infinite;
  }

  /* ── error ── */
  .qt-error {
    margin-top: 14px;
    background: rgba(239,68,68,.06);
    border: 1px solid rgba(239,68,68,.2);
    border-radius: 10px; padding: 13px 16px;
    font-size: .87rem; color: #fca5a5;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── results panel ── */
  .qt-results {
    background: rgba(14,18,32,.85);
    border: 1px solid rgba(74,222,128,.12);
    border-radius: 20px;
    padding: 32px 36px;
    backdrop-filter: blur(20px);
    animation: fadeUp .5s ease both;
  }

  .qt-results-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 24px;
  }

  .qt-results-label {
    font-size: .68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .14em;
    color: #4ade80;
  }

  .qt-results-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(74,222,128,.25), transparent);
  }

  /* ── question block ── */
  .qt-q {
    display: flex; gap: 14px;
    background: rgba(0,0,0,.25);
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 14px; padding: 18px 20px;
    margin-bottom: 10px;
    transition: border-color .2s, background .2s;
    cursor: default;
  }

  .qt-q:hover {
    border-color: rgba(56,189,248,.2);
    background: rgba(56,189,248,.03);
  }

  .qt-q-num {
    flex-shrink: 0; width: 32px; height: 32px;
    background: rgba(56,189,248,.1);
    border: 1px solid rgba(56,189,248,.25);
    color: #38bdf8; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem; font-weight: 600;
  }

  .qt-q-body { flex: 1; }

  .qt-q-text {
    font-size: .93rem; line-height: 1.7; color: #e2e8f0;
  }

  .qt-q-note {
    font-size: .78rem; color: #6366f1;
    font-style: italic; margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── meta bar ── */
  .qt-meta {
    display: flex; gap: 28px; flex-wrap: wrap;
    margin-top: 22px; padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,.05);
  }

  .qt-meta-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem; color: rgba(148,163,184,.4);
  }

  .qt-meta-val { color: #38bdf8; font-weight: 500; }
`;

/* ─── QuestionBlock ──────────────────────────────────────────────── */
function QuestionBlock({ index, raw }) {
  const noteMatch = raw.match(/\(([^)]+)\)\s*$/);
  const note = noteMatch ? noteMatch[1] : null;
  const body = raw.replace(/\([^)]+\)\s*$/, "").replace(/^\d+\.\s*/, "").trim();

  return (
    <div className="qt-q">
      <div className="qt-q-num">{String(index).padStart(2, "0")}</div>
      <div className="qt-q-body">
        <div className="qt-q-text">{body}</div>
        {note && <div className="qt-q-note">↳ {note}</div>}
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────── */
export default function QuestionTest() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef(null);

  // inject CSS once
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  async function handleGenerate() {
    if (!text.trim()) { setError("Please enter some transcribed text first."); return; }
    if (wordCount < 10) { setError("Text is too short — paste more content (at least 10 words)."); return; }

    setLoading(true);
    setError("");
    setResult(null);
    const t0 = Date.now();

    try {
      const res = await fetch(`${BACKEND_URL}/questions/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult({
        questions: data.questions,
        wordCount: data.word_count,
        elapsed: ((Date.now() - t0) / 1000).toFixed(1),
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function parseQuestions(raw) {
    const blocks = raw.split(/\n(?=\d+\.)/).map((s) => s.trim()).filter(Boolean);
    return blocks.length > 0 ? blocks : [raw];
  }

  return (
    <div className="qt-page">
      <style>{CSS}</style>
      <div className="qt-glow" />

      <div className="qt-content">

        {/* ── Header ── */}
        <div className="qt-header">
          <div className="qt-brand">
            <div className="qt-brand-icon">🎙</div>
            <span className="qt-brand-name">Auris</span>
          </div>
          <div className="qt-pill">
            <div className="qt-pill-dot" />
            qwen2.5:1.5b · Question Generator
          </div>
        </div>

        {/* ── Input Panel ── */}
        <div className="qt-panel">
          <div className="qt-panel-header">
            <span className="qt-panel-label">Transcribed Text Input</span>
            <div className="qt-panel-line" />
          </div>

          <div className={`qt-textarea-wrap ${focused ? "focused" : ""}`}>
            <textarea
              ref={textareaRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Paste your transcribed presentation text here..."
            />
            <div className="qt-char-count">
              {wordCount} words
            </div>
          </div>

          <div className="qt-btn-row">
            <button
              className="qt-btn-primary"
              onClick={handleGenerate}
              disabled={loading}
            >
              {loading ? "Generating…" : "⚡ Generate Questions"}
            </button>
            <button
              className="qt-btn-ghost"
              onClick={() => { setText(SAMPLE); setResult(null); setError(""); }}
            >
              Load sample
            </button>
            <button
              className="qt-btn-ghost"
              onClick={() => { setText(""); setResult(null); setError(""); }}
            >
              Clear
            </button>
          </div>

          {loading && (
            <div className="qt-status">
              <div className="qt-spinner" />
          ..
            </div>
          )}

          {error && <div className="qt-error">⚠ {error}</div>}
        </div>

        {/* ── Results Panel ── */}
        {result && (
          <div className="qt-results">
            <div className="qt-results-header">
              <span className="qt-results-label">Generated Jury-Style Questions</span>
              <div className="qt-results-line" />
            </div>

            {parseQuestions(result.questions).map((q, i) => (
              <QuestionBlock key={i} index={i + 1} raw={q} />
            ))}

            <div className="qt-meta">
              <div className="qt-meta-item">
                Model <span className="qt-meta-val">qwen2.5:1.5b</span>
              </div>
              <div className="qt-meta-item">
                Input <span className="qt-meta-val">{result.wordCount} words</span>
              </div>
              <div className="qt-meta-item">
                Time <span className="qt-meta-val">{result.elapsed}s</span>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}