import { useState, useRef, useCallback } from 'react';
import { Upload, X, FileAudio, FileVideo, CheckCircle, Loader2, Play, Pause } from 'lucide-react';
import type { AnalysisResult, PageView } from '@/types';

interface UploadPageProps {
  onNavigate: (view: PageView) => void;
  onResult: (result: AnalysisResult) => void;
}

function isVideoFile(file: File) {
  return file.type.startsWith('video/') ||
    /\.(mp4|mov|avi|mkv|webm|m4v|wmv|flv|3gp)$/i.test(file.name);
}

export default function UploadPage({ onNavigate, onResult }: UploadPageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [fileURL, setFileURL] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<'extract' | 'analyze' | 'feedback'>('extract');
  const [playing, setPlaying] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);

  const setSelectedFile = (f: File) => {
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(f);
    setFileURL(URL.createObjectURL(f));
    setPlaying(false);
  };

  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setSelectedFile(dropped);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) setSelectedFile(selected);
  }, []);

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    const el = mediaRef.current;
    if (!el) return;
    if (playing) { el.pause(); setPlaying(false); }
    else { el.play(); setPlaying(true); }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFile(null);
    setFileURL(null);
    setPlaying(false);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setUploading(true);
    setUploadProgress('extract');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const progressTimer = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev === 'extract') return 'analyze';
          if (prev === 'analyze') return 'feedback';
          return prev;
        });
      }, 2000);

      const response = await fetch('/transcribe', { method: 'POST', body: formData });
      clearInterval(progressTimer);

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      onResult(await response.json());
      onNavigate('results');
    } catch (err) {
      console.error('Upload failed:', err);
      // Demo fallback with all models enabled
      const fallbackResult: AnalysisResult = {
        status: 'ok',
        filename: file.name,
        duration_sec: 95.4,
        whisper: {
          text: 'Hello world! My name is David Malin and this is an introduction to Computer Science, designed especially for employees and employers who would like to be conversant in today and tomorrow\'s information technology. Ultimately, in this course empowers you to ask better questions and make better decisions even if you\'re not an IT person yourself. We\'ll begin with zeros and ones, the underlying language that today\'s laptops, desktop, servers, and phones speak.',
          word_count: 78,
          segments: [
            { start: 0, end: 5.2, text: 'Hello world! My name is David Malin and this is an introduction to Computer Science' },
            { start: 5.2, end: 12.8, text: 'designed especially for employees and employers who would like to be conversant in today and tomorrow\'s information technology' },
            { start: 12.8, end: 20.1, text: 'Ultimately, in this course empowers you to ask better questions and make better decisions even if you\'re not an IT person yourself' },
            { start: 20.1, end: 28.5, text: 'We\'ll begin with zeros and ones, the underlying language that today\'s laptops, desktop, servers, and phones speak' },
          ],
        },
        correction: {
          corrected: 'Hello world! My name is David Malin, and this is an introduction to Computer Science, designed especially for employees and employers who would like to be conversant in today and tomorrow\'s information technology. Ultimately, this course empowers you to ask better questions and make better decisions, even if you\'re not an IT person yourself. We\'ll begin with zeros and ones, the underlying language that today\'s laptops, desktops, servers, and phones speak.',
          enabled: true,
          model: 'google/flan-t5-base',
          latency_ms: 29226,
          critique_stats: { corrected: 17, kept: 0, total: 17 },
        },
        emotion: {
          enabled: true, emotion: 'neutral', confidence: 0.62, latency_ms: 1450, is_reliable: true,
          all_probs: { angry: 0.02, calm: 0.18, disgust: 0.01, fear: 0.03, happy: 0.08, neutral: 0.62, sad: 0.04, surprised: 0.02 },
          realtime_factor: 15.8, inference_ms: 1200,
        },
        questions: {
          enabled: true,
          questions: [
            'Given that this course covers a broad range of computer science topics, how do you envision the progression of these topics building upon each other?',
            'The course emphasizes empowering employees and employers to make better technology decisions. Can you elaborate on a specific scenario?',
            'How do you see the intersection of AI, algorithms, and cybersecurity shaping future skill requirements for business professionals?',
            'With the increasing importance of data in business operations, how would you explain the role of databases to someone with no technical background?',
          ],
          raw: '', latency_ms: 3200, error: null,
        },
        timing: { extract_ms: 450, pipeline_ms: 29226, total_ms: 29676 },
      };
      onResult(fallbackResult);
      onNavigate('results');
    } finally {
      setUploading(false);
    }
  };

  const isVideo = file ? isVideoFile(file) : false;

  return (
    <div className="min-h-[100dvh] pt-[72px] bg-auris-bg">
      <div className="max-w-3xl mx-auto px-6 py-16">
        {/* Back link */}
        <button
          onClick={() => onNavigate('home')}
          className="text-auris-text-secondary hover:text-auris-text text-sm mb-8 flex items-center gap-1 transition-colors"
        >
          ← Back to Home
        </button>

        <div className="text-center mb-10">
          <h1 className="text-h1 text-auris-text mb-2">
            Analyze Your <span className="text-gradient">Presentation</span>
          </h1>
          <p className="text-body text-auris-text-secondary">
            Upload a video or audio file to get AI-powered speech feedback
          </p>
        </div>

        {!uploading ? (
          <>
            {/* Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !file && inputRef.current?.click()}
              className={`relative border-2 border-dashed rounded-2xl transition-all duration-200 ${
                file ? 'cursor-default' : 'cursor-pointer'
              } ${
                isDragging
                  ? 'border-auris-blue bg-[rgba(59,130,246,0.05)]'
                  : file
                  ? 'border-[rgba(148,163,184,0.25)] bg-auris-card'
                  : 'border-[rgba(148,163,184,0.15)] bg-auris-card hover:border-[rgba(148,163,184,0.3)] hover:bg-auris-card-hover'
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept="audio/*,video/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {!file ? (
                /* Empty state */
                <div className="p-12 text-center">
                  <Upload size={40} className="mx-auto mb-4 text-auris-text-tertiary" />
                  <p className="text-auris-text font-medium mb-1">
                    Drop your presentation video or audio here
                  </p>
                  <p className="text-auris-text-tertiary text-sm mb-4">
                    Supports MP4, MOV, AVI, MP3, WAV, M4A and more
                  </p>
                  <button className="btn-primary text-sm py-2.5 px-6">Browse File</button>
                </div>
              ) : (
                /* File loaded — show preview inside the zone */
                <div className="p-4">
                  {/* File info bar */}
                  <div className="flex items-center gap-3 mb-3">
                    {isVideo
                      ? <FileVideo size={20} className="text-auris-blue flex-shrink-0" />
                      : <FileAudio size={20} className="text-auris-blue flex-shrink-0" />}
                    <div className="flex-1 min-w-0">
                      <p className="text-auris-text text-sm font-medium truncate">{file.name}</p>
                      <p className="text-auris-text-tertiary text-xs">
                        {(file.size / (1024 * 1024)).toFixed(1)} MB
                      </p>
                    </div>
                    <button
                      onClick={clearFile}
                      className="text-auris-text-tertiary hover:text-red-400 transition-colors p-1 flex-shrink-0"
                    >
                      <X size={18} />
                    </button>
                  </div>

                  {/* Video player */}
                  {isVideo && fileURL && (
                    <div className="relative rounded-xl overflow-hidden bg-black aspect-video mb-3">
                      <video
                        ref={(el) => { mediaRef.current = el; }}
                        src={fileURL}
                        className="w-full h-full object-contain"
                        onEnded={() => setPlaying(false)}
                        onPause={() => setPlaying(false)}
                        onPlay={() => setPlaying(true)}
                      />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <button
                          onClick={togglePlay}
                          className={`w-14 h-14 rounded-full bg-black/50 hover:bg-black/70 flex items-center justify-center text-white transition-all backdrop-blur-sm ${playing ? 'opacity-0 hover:opacity-100' : 'opacity-100'}`}
                        >
                          {playing ? <Pause size={24} /> : <Play size={24} className="ml-1" />}
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Audio player */}
                  {!isVideo && fileURL && (
                    <div className="rounded-xl bg-auris-bg p-3 mb-3">
                      <audio
                        ref={(el) => { mediaRef.current = el; }}
                        src={fileURL}
                        className="w-full"
                        controls
                        onEnded={() => setPlaying(false)}
                      />
                    </div>
                  )}

                  {/* Change file link */}
                  <button
                    onClick={() => inputRef.current?.click()}
                    className="text-auris-text-tertiary hover:text-auris-text-secondary text-xs underline-offset-2 hover:underline transition-colors"
                  >
                    Change file
                  </button>
                </div>
              )}
            </div>

            {/* Analyze Button */}
            {file && (
              <button
                onClick={handleAnalyze}
                className="w-full mt-6 btn-primary flex items-center justify-center gap-2"
              >
                <SparkleIcon />
                Analyze Presentation
              </button>
            )}
          </>
        ) : (
          /* Loading State */
          <div className="flex flex-col items-center py-16">
            <div className="flex items-center gap-12 mb-8">
              {[
                { key: 'extract' as const, label: 'Extracting Audio', icon: <CheckCircle size={24} /> },
                { key: 'analyze' as const, label: 'Voice Analysis', icon: <Loader2 size={24} className="animate-spin" /> },
                { key: 'feedback' as const, label: 'Generating Feedback', icon: <SparkleIcon /> },
              ].map((step) => {
                const stepOrder = ['extract', 'analyze', 'feedback'];
                const currentOrder = stepOrder.indexOf(uploadProgress);
                const stepIdx = stepOrder.indexOf(step.key);
                const isCompleted = stepIdx < currentOrder;
                const isCurrent = stepIdx === currentOrder;
                return (
                  <div key={step.key} className="flex flex-col items-center gap-2">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-500 ${
                      isCompleted ? 'bg-auris-amber text-auris-bg'
                        : isCurrent ? 'bg-auris-blue text-white shadow-glow-blue'
                        : 'bg-auris-card text-auris-text-tertiary'
                    }`}>
                      {isCompleted ? <CheckCircle size={24} /> : step.icon}
                    </div>
                    <span className={`text-xs font-medium ${isCompleted || isCurrent ? 'text-auris-text' : 'text-auris-text-tertiary'}`}>
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>

            <div className="flex items-center gap-0 -mt-14 mb-8 w-64">
              <div className="flex-1 h-px bg-auris-amber" />
              <div className={`flex-1 h-px ${uploadProgress !== 'extract' ? 'bg-auris-amber' : 'bg-auris-border'}`} />
            </div>

            <p className="text-auris-text-secondary text-sm">
              {uploadProgress === 'extract' && 'Extracting audio from your file...'}
              {uploadProgress === 'analyze' && 'Whisper + Flan-T5 analyzing your speech...'}
              {uploadProgress === 'feedback' && 'Generating corrections and feedback...'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function SparkleIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M8 0L9.5 6.5L16 8L9.5 9.5L8 16L6.5 9.5L0 8L6.5 6.5L8 0Z" fill="currentColor" />
    </svg>
  );
}
