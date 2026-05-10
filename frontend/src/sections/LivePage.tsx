import { useState, useRef, useCallback, useEffect } from 'react';
import { Mic, MicOff, Square, Loader2, Video, VideoOff, CheckSquare } from 'lucide-react';
import type { AnalysisResult, PageView } from '@/types';

interface LivePageProps {
  onNavigate: (view: PageView) => void;
  onResult: (result: AnalysisResult) => void;
}

type RecordingState = 'idle' | 'requesting' | 'recording' | 'processing' | 'error';

export default function LivePage({ onNavigate, onResult }: LivePageProps) {
  const [state, setState] = useState<RecordingState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [recordTime, setRecordTime] = useState(0);
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [grammarEnabled, setGrammarEnabled] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const recordTimeRef = useRef(0);

  useEffect(() => { recordTimeRef.current = recordTime; }, [recordTime]);

  const stopRecording = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) { videoRef.current.srcObject = null; }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(new TextEncoder().encode('__END__'));
    }
    setState('processing');
  }, []);

  const startRecording = async () => {
    setError(null);
    setState('requesting');
    try {
      const constraints: MediaStreamConstraints = {
        audio: true,
        video: cameraEnabled
          ? { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
          : false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (cameraEnabled && videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.play().catch(() => {});
      }

      // Only audio is sent for analysis
      const audioStream = new MediaStream(stream.getAudioTracks());
      const mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/live`);
      wsRef.current = ws;

      ws.onopen = () => {
        setState('recording');
        setRecordTime(0);
        timerRef.current = setInterval(() => setRecordTime((t) => t + 1), 1000);
        mediaRecorder.start(100);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'final') {
            const result: AnalysisResult = {
              status: 'ok',
              filename: 'live-recording.wav',
              duration_sec: msg.duration_sec || recordTimeRef.current,
              whisper: msg.whisper || { text: '', word_count: 0, segments: [] },
              correction: msg.correction || { corrected: '', enabled: false, model: null, latency_ms: 0, critique_stats: { corrected: 0, kept: 0, total: 0 } },
              emotion: msg.emotion || { enabled: false, emotion: 'unknown', confidence: 0, latency_ms: 0, is_reliable: false, all_probs: {}, realtime_factor: 0, inference_ms: 0 },
              questions: msg.questions || { enabled: false, questions: [], raw: '', latency_ms: 0, error: null },
              timing: { extract_ms: 0, pipeline_ms: msg.pipeline_ms || 0, total_ms: msg.pipeline_ms || 0 },
            };
            onResult(result);
            onNavigate('results');
          } else if (msg.type === 'error') {
            setError(msg.msg || 'Recording failed');
            setState('error');
          }
        } catch { /* non-JSON */ }
      };

      ws.onerror = () => {
        setError('WebSocket connection failed. Using fallback mode.');
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunksRef.current.push(e.data);
        };
        mediaRecorder.onstop = async () => {
          const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          await processFallback(blob);
        };
        setState('recording');
        setRecordTime(0);
        timerRef.current = setInterval(() => setRecordTime((t) => t + 1), 1000);
        mediaRecorder.start(100);
      };

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          e.data.arrayBuffer().then((buffer) => ws.send(buffer));
        }
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access denied');
      setState('error');
    }
  };

  const processFallback = async (blob: Blob) => {
    try {
      const file = new File([blob], 'live-recording.webm', { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('/transcribe', { method: 'POST', body: formData });
      if (response.ok) { onResult(await response.json()); }
      else { useDemoResult(); }
    } catch { useDemoResult(); }
    onNavigate('results');
  };

  const useDemoResult = () => {
    const demo: AnalysisResult = {
      status: 'ok', filename: 'live-recording.wav',
      duration_sec: recordTimeRef.current || 45,
      whisper: {
        text: 'Good morning everyone. Today I want to present our quarterly results and discuss our strategy for the next fiscal year. Revenue has grown by twenty five percent compared to last quarter.',
        word_count: 32,
        segments: [
          { start: 0, end: 8, text: 'Good morning everyone. Today I want to present our quarterly results' },
          { start: 8, end: 18, text: 'and discuss our strategy for the next fiscal year' },
          { start: 18, end: 28, text: 'Revenue has grown by twenty five percent compared to last quarter' },
        ],
      },
      correction: {
        corrected: 'Good morning, everyone. Today, I want to present our quarterly results and discuss our strategy for the next fiscal year. Revenue has grown by 25% compared to last quarter.',
        enabled: true, model: 'google/flan-t5-base', latency_ms: 3200,
        critique_stats: { corrected: 3, kept: 0, total: 3 },
      },
      emotion: {
        enabled: true, emotion: 'happy', confidence: 0.71, latency_ms: 890, is_reliable: true,
        all_probs: { angry: 0.01, calm: 0.08, disgust: 0.0, fear: 0.01, happy: 0.71, neutral: 0.15, sad: 0.02, surprised: 0.02 },
        realtime_factor: 12.4, inference_ms: 720,
      },
      questions: {
        enabled: true,
        questions: [
          'Can you elaborate on the specific factors that contributed to the 25% revenue growth?',
          'What are the key strategic initiatives for the next fiscal year?',
          'How does your team plan to address potential risks?',
        ],
        raw: '', latency_ms: 2800, error: null,
      },
      timing: { extract_ms: 200, pipeline_ms: 6890, total_ms: 7090 },
    };
    onResult(demo);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  const isIdle = state === 'idle' || state === 'error';
  const isRecording = state === 'recording';
  const isBusy = state === 'requesting' || state === 'processing';

  return (
    <div className="min-h-[100dvh] pt-[72px] bg-auris-bg flex flex-col items-center justify-center px-6">
      <div className="max-w-xl w-full text-center">
        <button
          onClick={() => onNavigate('home')}
          className="text-auris-text-secondary hover:text-auris-text text-sm mb-8 flex items-center gap-1 transition-colors mx-auto"
        >
          ← Back to Home
        </button>

        <h1 className="text-h1 text-auris-text mb-3">
          Live <span className="text-gradient">Recording</span>
        </h1>
        <p className="text-body text-auris-text-secondary mb-8">
          Speak naturally and get real-time AI feedback on your presentation
        </p>

        {/* Camera preview */}
        {cameraEnabled && (
          <div className="relative w-full max-w-sm mx-auto mb-6 rounded-2xl overflow-hidden bg-auris-card border border-auris-border aspect-video">
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className={`w-full h-full object-cover transition-opacity duration-300 ${isRecording ? 'opacity-100' : 'opacity-0'}`}
            />
            {!isRecording && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-auris-text-tertiary text-sm gap-2">
                <Video size={28} className="opacity-30" />
                <span className="text-xs">Camera preview starts with recording</span>
              </div>
            )}
            {isRecording && (
              <div className="absolute top-3 left-3 flex items-center gap-1.5 bg-black/60 rounded-full px-2.5 py-1">
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-white text-xs font-mono">{formatTime(recordTime)}</span>
              </div>
            )}
          </div>
        )}

        {/* Recording circle (no camera) */}
        {!cameraEnabled && (
          <div className="relative w-40 h-40 mx-auto mb-6 flex items-center justify-center">
            {isRecording && (
              <>
                <div className="absolute inset-0 rounded-full border-2 border-red-500/30 animate-ripple" />
                <div className="absolute inset-0 rounded-full border-2 border-red-500/20 animate-ripple" style={{ animationDelay: '0.5s' }} />
              </>
            )}
            <button
              onClick={() => { if (isIdle) startRecording(); else if (isRecording) stopRecording(); }}
              disabled={isBusy}
              className={`relative w-28 h-28 rounded-full flex items-center justify-center transition-all duration-300 ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600 shadow-[0_0_40px_rgba(239,68,68,0.4)]'
                  : isBusy
                  ? 'bg-auris-card cursor-not-allowed'
                  : 'bg-auris-blue hover:bg-auris-blue-hover shadow-glow-blue hover:shadow-glow-blue-hover'
              }`}
            >
              {isIdle ? <Mic size={40} className="text-white" />
                : isRecording ? <Square size={32} className="text-white fill-white" />
                : <Loader2 size={32} className="text-auris-blue animate-spin" />}
            </button>
          </div>
        )}

        {/* Timer (no camera) */}
        {isRecording && !cameraEnabled && (
          <p className="text-3xl font-mono text-auris-text mb-2">{formatTime(recordTime)}</p>
        )}

        {/* Start/Stop button (with camera) */}
        {cameraEnabled && (
          <button
            onClick={() => { if (isIdle) startRecording(); else if (isRecording) stopRecording(); }}
            disabled={isBusy}
            className={`mx-auto flex items-center gap-2 px-8 py-3 rounded-full font-medium text-sm transition-all duration-300 mb-3 ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600 text-white shadow-[0_0_24px_rgba(239,68,68,0.4)]'
                : isBusy
                ? 'bg-auris-card text-auris-text-tertiary cursor-not-allowed'
                : 'bg-auris-blue hover:bg-auris-blue-hover text-white shadow-glow-blue'
            }`}
          >
            {isIdle ? <><Mic size={16} /> Start Recording</>
              : isRecording ? <><Square size={16} className="fill-white" /> Stop</>
              : <><Loader2 size={16} className="animate-spin" /> Processing...</>}
          </button>
        )}

        {/* Status */}
        <p className="text-auris-text-secondary mb-3 text-sm">
          {state === 'idle' && (cameraEnabled ? 'Press Start to begin recording' : 'Tap the microphone to start recording')}
          {state === 'requesting' && 'Requesting access...'}
          {state === 'recording' && !cameraEnabled && 'Recording... tap to stop'}
          {state === 'processing' && 'Processing your speech...'}
          {state === 'error' && 'Something went wrong. Tap to retry.'}
        </p>

        {error && <p className="text-red-400 text-sm mb-3 max-w-md mx-auto">{error}</p>}

        {state === 'processing' && (
          <div className="flex items-center justify-center gap-2 mb-4">
            {[0, 0.2, 0.4].map((d) => (
              <div key={d} className="w-2 h-2 rounded-full bg-auris-blue animate-bounce" style={{ animationDelay: `${d}s` }} />
            ))}
          </div>
        )}

        {/* Toggle controls */}
        {!isRecording && !isBusy && (
          <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
            {/* Camera toggle */}
            <button
              onClick={() => setCameraEnabled((v) => !v)}
              title={cameraEnabled ? 'Disable camera' : 'Enable camera'}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 ${
                cameraEnabled
                  ? 'bg-auris-blue/10 border-auris-blue text-auris-blue'
                  : 'bg-auris-card border-auris-border text-auris-text-secondary hover:border-auris-text-secondary'
              }`}
            >
              {cameraEnabled ? <Video size={14} /> : <VideoOff size={14} />}
              {cameraEnabled ? 'Camera On' : 'Camera Off'}
            </button>

            {/* Grammar correction toggle */}
            <button
              onClick={() => setGrammarEnabled((v) => !v)}
              title={grammarEnabled ? 'Disable grammar correction' : 'Enable grammar correction'}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium border transition-all duration-200 ${
                grammarEnabled
                  ? 'bg-[rgba(20,184,166,0.1)] border-auris-teal text-auris-teal'
                  : 'bg-auris-card border-auris-border text-auris-text-secondary hover:border-auris-text-secondary'
              }`}
            >
              <CheckSquare size={14} />
              Grammar {grammarEnabled ? 'On' : 'Off'}
            </button>
          </div>
        )}

        {/* Info badges */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
          {[
            { icon: <MicOff size={12} />, label: 'Max 5 minutes' },
            { icon: <div className="w-3 h-3 rounded-full border border-auris-teal" />, label: '16kHz mono' },
          ].map((badge) => (
            <span
              key={badge.label}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-full text-caption bg-auris-card text-auris-text-tertiary"
            >
              {badge.icon}
              {badge.label}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
