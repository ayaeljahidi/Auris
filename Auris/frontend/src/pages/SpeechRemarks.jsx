import { useState, useEffect, useRef } from "react";

const BACKEND_URL = "http://localhost:8000";
const MODEL_NAME = "qwen2.5:7b";

const SAMPLE_SPEECHES = [
  {
    label: "Group project",
    text: `Good morning. Today I will present our project called DataFlow. It is a real-time data pipeline that processes large amounts of data. I built the ingestion layer using Apache Kafka. It handles the incoming data streams. Then my teammate worked on the processing part but I will explain it. The processing uses Spark and it transforms the data. I think it works well. Sarah already talked about the storage part but basically I designed the database schema myself. We used PostgreSQL. The system stores processed data and users can query it. The results were good. We tested it and it handled the load. If we had more time I would add more features to my part. Thank you.`
  },
  {
    label: "Unclear flow",
    text: `Our project is about security. Encryption is important. We implemented it. The database has tables and relationships. Users can log in. The frontend is in React. We also have an API. Docker is used. The system is deployed. We faced challenges. We solved them. The project works. Performance is acceptable. Users liked it. It is secure and fast and reliable and scalable.`
  },
  {
    label: "Hollow speech",
    text: `Hello everyone, so basically today we are going to talk about our amazing project SmartCampus. It is kind of like a smart campus management system that does a lot of things. The problem is that campus management is inefficient you know. Our solution fixes all of these problems. We used AI and machine learning and cloud computing and IoT and blockchain to make it better. The AI part is really intelligent and learns from the data. The IoT sensors collect information from everywhere. The blockchain ensures trust. The cloud makes it scalable. We tested it and everything worked perfectly. Users were very satisfied. The system is very very good and we are very proud of what we did. In conclusion our project is revolutionary and will change how campuses are managed forever. Thank you so much.`
  }
];

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  @keyframes fadeUp    { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:none; } }
  @keyframes spin      { to { transform: rotate(360deg); } }
  @keyframes scanline  { 0%,100% { opacity:.03; } 50% { opacity:.07; } }
  @keyframes glow-pulse { 0%,100% { box-shadow: 0 0 20px rgba(248,113,113,.15); }
                          50%     { box-shadow: 0 0 42px rgba(248,113,113,.35); } }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #060912; }

  .rm-page {
    min-height: 100vh; background: #060912;
    font-family: 'Syne', sans-serif; color: #cdd6f4;
    display: flex; flex-direction: column; align-items: center;
    padding: 56px 20px 100px; position: relative; overflow-x: hidden;
  }
  .rm-page::before {
    content: ''; position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px,
      rgba(255,255,255,.013) 2px, rgba(255,255,255,.013) 4px);
    pointer-events: none; z-index: 0; animation: scanline 4s ease-in-out infinite;
  }
  .rm-glow {
    position: fixed; top: -180px; left: 50%; transform: translateX(-50%);
    width: 900px; height: 500px;
    background: radial-gradient(ellipse at center,
      rgba(248,113,113,.07) 0%, rgba(192,132,252,.05) 40%, transparent 70%);
    pointer-events: none; z-index: 0;
  }
  .rm-content { position: relative; z-index: 1; width: 100%; max-width: 820px; }

  .rm-header { text-align: center; margin-bottom: 52px; animation: fadeUp .6s ease both; }
  .rm-brand { display: inline-flex; align-items: center; gap: 12px; margin-bottom: 18px; }
  .rm-brand-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #f87171, #c084fc);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 0 28px rgba(248,113,113,.35);
  }
  .rm-brand-name {
    font-size: 2rem; font-weight: 800; letter-spacing: -.04em;
    background: linear-gradient(135deg, #e2e8f0 30%, #f87171 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .rm-pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(248,113,113,.07); border: 1px solid rgba(248,113,113,.18);
    border-radius: 999px; padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem; color: #f87171; letter-spacing: .06em;
  }
  .rm-pill-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #f87171; box-shadow: 0 0 8px #f87171;
  }

  .rm-panel {
    background: rgba(14,18,32,.85); border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px; padding: 32px 36px; margin-bottom: 18px;
    backdrop-filter: blur(20px); animation: fadeUp .6s ease both;
  }
  .rm-panel-header { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
  .rm-panel-label {
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .14em; color: #f87171;
  }
  .rm-panel-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(248,113,113,.25), transparent);
  }

  .rm-samples { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  .rm-sample-btn {
    padding: 5px 12px; border-radius: 8px; cursor: pointer;
    font-family: 'JetBrains Mono', monospace; font-size: .72rem;
    border: 1px solid rgba(248,113,113,.2);
    background: rgba(248,113,113,.05); color: rgba(248,113,113,.7);
    transition: all .2s;
  }
  .rm-sample-btn:hover { background: rgba(248,113,113,.12); color: #f87171; border-color: rgba(248,113,113,.4); }

  .rm-textarea-wrap {
    position: relative; border: 1px solid rgba(248,113,113,.18);
    border-radius: 14px; overflow: hidden; transition: border-color .2s, box-shadow .2s;
  }
  .rm-textarea-wrap.focused {
    border-color: rgba(248,113,113,.5);
    box-shadow: 0 0 0 3px rgba(248,113,113,.08), inset 0 0 30px rgba(248,113,113,.03);
  }
  .rm-textarea-wrap textarea {
    width: 100%; min-height: 200px; background: rgba(6,9,18,.9);
    border: none; outline: none; color: #cdd6f4;
    font-family: 'JetBrains Mono', monospace; font-size: .88rem;
    line-height: 1.75; padding: 18px 20px 36px; resize: vertical;
    caret-color: #f87171;
  }
  .rm-textarea-wrap textarea::placeholder { color: rgba(148,163,184,.35); }
  .rm-char-count {
    position: absolute; bottom: 10px; right: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem; color: rgba(148,163,184,.4); pointer-events: none;
  }

  .rm-btn-row { display: flex; gap: 10px; margin-top: 16px; }
  .rm-btn-primary {
    flex: 1; padding: 14px;
    background: linear-gradient(135deg, #f87171, #c084fc);
    color: #fff; border: none; border-radius: 12px;
    font-family: 'Syne', sans-serif; font-size: .95rem; font-weight: 700;
    cursor: pointer; box-shadow: 0 4px 24px rgba(248,113,113,.25);
    transition: opacity .2s, transform .1s;
    animation: glow-pulse 3s ease-in-out infinite;
  }
  .rm-btn-primary:hover:not(:disabled) { opacity: .92; transform: translateY(-1px); }
  .rm-btn-primary:active:not(:disabled) { transform: translateY(0); }
  .rm-btn-primary:disabled {
    background: rgba(248,113,113,.12); color: rgba(148,163,184,.4);
    cursor: not-allowed; box-shadow: none; animation: none;
  }
  .rm-btn-ghost {
    padding: 14px 18px; background: transparent; color: rgba(148,163,184,.7);
    border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
    font-family: 'Syne', sans-serif; font-size: .85rem; cursor: pointer;
    transition: border-color .2s, color .2s; white-space: nowrap;
  }
  .rm-btn-ghost:hover { border-color: rgba(248,113,113,.3); color: #f87171; }

  .rm-status {
    display: flex; align-items: center; gap: 12px; margin-top: 16px;
    background: rgba(248,113,113,.05); border: 1px solid rgba(248,113,113,.12);
    border-radius: 10px; padding: 13px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: #f87171;
  }
  .rm-spinner {
    width: 16px; height: 16px; flex-shrink: 0;
    border: 2px solid rgba(248,113,113,.15); border-top-color: #f87171;
    border-radius: 50%; animation: spin .75s linear infinite;
  }
  .rm-error {
    margin-top: 14px; background: rgba(239,68,68,.06);
    border: 1px solid rgba(239,68,68,.2); border-radius: 10px;
    padding: 13px 16px; font-size: .87rem; color: #fca5a5;
    font-family: 'JetBrains Mono', monospace;
  }

  .rm-results {
    background: rgba(14,18,32,.85); border: 1px solid rgba(248,113,113,.12);
    border-radius: 20px; padding: 32px 36px;
    backdrop-filter: blur(20px); animation: fadeUp .5s ease both;
  }
  .rm-results-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 24px;
  }
  .rm-results-label {
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .14em; color: #f87171;
  }
  .rm-results-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(248,113,113,.25), transparent);
  }
  .rm-raw-btn {
    background: transparent; border: 1px solid rgba(248,113,113,.2);
    border-radius: 8px; padding: 4px 10px; cursor: pointer;
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    color: rgba(248,113,113,.6); white-space: nowrap; transition: all .2s;
  }
  .rm-raw-btn:hover { border-color: rgba(248,113,113,.5); color: #f87171; }

  .rm-raw {
    font-family: 'JetBrains Mono', monospace; font-size: .78rem;
    color: rgba(205,214,244,.5); line-height: 1.7;
    white-space: pre-wrap; word-break: break-word;
    background: rgba(0,0,0,.3); border-radius: 10px; padding: 14px 16px;
  }

  .rm-item {
    padding: 16px 18px; border-radius: 14px; margin-bottom: 10px;
    background: rgba(248,113,113,.05); border: 1px solid rgba(248,113,113,.15);
    transition: border-color .2s, background .2s;
  }
  .rm-item:hover { background: rgba(248,113,113,.08); border-color: rgba(248,113,113,.25); }
  .rm-item:last-child { margin-bottom: 0; }
  .rm-item-num {
    font-family: 'JetBrains Mono', monospace; font-size: .68rem;
    color: rgba(248,113,113,.5); margin-bottom: 8px; font-weight: 600;
  }
  .rm-quote {
    font-family: 'JetBrains Mono', monospace; font-size: .82rem;
    color: rgba(205,214,244,.5); margin-bottom: 8px; line-height: 1.6;
    border-left: 2px solid rgba(248,113,113,.3); padding-left: 10px;
  }
  .rm-problem { font-size: .9rem; color: #e2e8f0; line-height: 1.65; }

  .rm-empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: .85rem; color: rgba(248,113,113,.5); padding: 12px 0;
  }

  .rm-meta {
    display: flex; gap: 28px; flex-wrap: wrap; margin-top: 22px;
    padding-top: 18px; border-top: 1px solid rgba(255,255,255,.05);
  }
  .rm-meta-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: .75rem; color: rgba(148,163,184,.4);
  }
  .rm-meta-val { color: #f87171; font-weight: 500; }
`;
function parseRemarks(raw) {
  if (!raw) return [];
  const results = [];

  // strategy 1 — ❌ CATEGORY: "quote" format
  const blocks = raw.split(/(?=❌)/).map(b => b.trim()).filter(Boolean);
  if (blocks.length > 0 && blocks[0].startsWith('❌')) {
    for (const block of blocks) {
      const categoryMatch = block.match(/❌\s*([A-Z]+)\s*:/);
      const quoteMatch = block.match(/:\s*["""]([^"""]+)["""]/);
      const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
      const problem = lines.find(l => !l.startsWith('❌') && l.length > 8);
      if (quoteMatch && problem) {
        results.push({
          category: categoryMatch ? categoryMatch[1] : 'REMARK',
          quote: quoteMatch[1].trim(),
          problem,
        });
      }
    }
    if (results.length > 0) return results;
  }

  // strategy 2 — numbered list fallback
  const numbered = raw.match(/\d+\.\s+.+/g);
  if (numbered) {
    for (const line of numbered) {
      const clean = line.replace(/^\d+\.\s*/, '');
      const parts = clean.split(/\s*[—–-]{1,2}\s*/);
      if (parts.length >= 2) {
        results.push({
          category: 'REMARK',
          quote: parts[0].replace(/["""]/g, '').trim(),
          problem: parts.slice(1).join(' ').trim(),
        });
      }
    }
    if (results.length > 0) return results;
  }

  // strategy 3 — alternate lines fallback
  const lines = raw.split('\n').map(l => l.trim()).filter(l => l.length > 8);
  for (let i = 0; i + 1 < lines.length; i += 2) {
    results.push({
      category: 'REMARK',
      quote: lines[i].replace(/^[❌\-\d\.\s]*/, '').replace(/["""]/g, '').trim(),
      problem: lines[i + 1].replace(/^[❌\-\d\.\s]*/, '').trim(),
    });
  }

  return results;
}
const CATEGORY_COLORS = {
  MEANING:   '#38bdf8',
  FLOW:      '#fb923c',
  CLARITY:   '#a78bfa',
  COHESION:  '#34d399',
  LANGUAGE:  '#f87171',
  STRUCTURE: '#fbbf24',
  REMARK:    '#94a3b8',
};

function RemarkItem({ index, remark }) {
  const color = CATEGORY_COLORS[remark.category] || CATEGORY_COLORS.REMARK;
  return (
    <div className="rm-item">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '.68rem',
          fontWeight: 600, color, letterSpacing: '.08em',
        }}>
          {remark.category}
        </span>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontSize: '.65rem',
          color: 'rgba(148,163,184,.3)',
        }}>
          #{String(index).padStart(2, '0')}
        </span>
      </div>
      <div className="rm-quote">"{remark.quote}"</div>
      <div className="rm-problem">{remark.problem}</div>
    </div>
  );
} 
export default function SpeechRemarks() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [focused, setFocused] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const textareaRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;

  async function handleAnalyze() {
    if (!text.trim()) { setError('Please enter a speech first.'); return; }
    if (wordCount < 20) { setError('Text is too short — need at least 20 words.'); return; }

    setLoading(true);
    setError('');
    setResult(null);
    setShowRaw(false);
    const t0 = Date.now();

    try {
      const res = await fetch(`${BACKEND_URL}/remarks/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      setResult({
        remarks: data.remarks,
        wordCount: data.word_count,
        elapsed: ((Date.now() - t0) / 1000).toFixed(1),
      });
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function loadSample(t) { setText(t); setResult(null); setError(''); setShowRaw(false); }
  function clear() { setText(''); setResult(null); setError(''); setShowRaw(false); }

  const parsed = result ? parseRemarks(result.remarks) : [];

  return (
    <div className="rm-page">
      <style>{CSS}</style>
      <div className="rm-glow" />
      <div className="rm-content">

        {/* Header */}
        <div className="rm-header">
          <div className="rm-brand">
            <div className="rm-brand-icon">🔍</div>
            <span className="rm-brand-name">Auris</span>
          </div>
          <div className="rm-pill">
            <div className="rm-pill-dot" />
            {MODEL_NAME} · Speech Coach
          </div>
        </div>

        {/* Input panel */}
        <div className="rm-panel">
          <div className="rm-panel-header">
            <span className="rm-panel-label">Speech Transcript</span>
            <div className="rm-panel-line" />
          </div>

          <div className="rm-samples">
            {SAMPLE_SPEECHES.map((s, i) => (
              <button key={i} className="rm-sample-btn" onClick={() => loadSample(s.text)}>
                {s.label}
              </button>
            ))}
          </div>

          <div className={`rm-textarea-wrap ${focused ? 'focused' : ''}`}>
            <textarea
              ref={textareaRef}
              value={text}
              onChange={e => setText(e.target.value)}
              onFocus={() => setFocused(true)}
              onBlur={() => setFocused(false)}
              placeholder="Paste your presentation speech here..."
            />
            <div className="rm-char-count">{wordCount} words</div>
          </div>

          <div className="rm-btn-row">
            <button className="rm-btn-primary" onClick={handleAnalyze} disabled={loading}>
              {loading ? 'Analyzing…' : '🔍 Analyze Speech'}
            </button>
            <button className="rm-btn-ghost" onClick={clear}>Clear</button>
          </div>

          {loading && (
            <div className="rm-status">
              <div className="rm-spinner" />
              Analyzing your speech…
            </div>
          )}
          {error && <div className="rm-error">⚠ {error}</div>}
        </div>

        {/* Results */}
        {result && (
          <div className="rm-results" ref={resultsRef}>
            <div className="rm-results-header">
              <span className="rm-results-label">
                {parsed.length} Remark{parsed.length !== 1 ? 's' : ''} Found
              </span>
              <div className="rm-results-line" />
              <button className="rm-raw-btn" onClick={() => setShowRaw(v => !v)}>
                {showRaw ? 'structured' : 'raw'}
              </button>
            </div>

            {showRaw ? (
              <pre className="rm-raw">{result.remarks}</pre>
            ) : parsed.length > 0 ? (
              parsed.map((r, i) => <RemarkItem key={i} index={i + 1} remark={r} />)
            ) : (
              <div className="rm-empty">
                Could not parse remarks — click "raw" to see the model output and share it so we can fix the parser.
              </div>
            )}

            <div className="rm-meta">
              <div className="rm-meta-item">Model <span className="rm-meta-val">{MODEL_NAME}</span></div>
              <div className="rm-meta-item">Input <span className="rm-meta-val">{result.wordCount} words</span></div>
              <div className="rm-meta-item">Time <span className="rm-meta-val">{result.elapsed}s</span></div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}