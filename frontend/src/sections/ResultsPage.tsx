import { useState } from 'react';
import {
  FileText, Sparkles, Smile, HelpCircle, Clock,
  ArrowLeft, Copy, Check, ChevronDown, BarChart3,
  Mic, Activity, Languages, Zap
} from 'lucide-react';
import type { AnalysisResult, PageView } from '@/types';

interface ResultsPageProps {
  result: AnalysisResult;
  onNavigate: (view: PageView) => void;
}

type TabId = 'transcript' | 'correction' | 'emotion' | 'questions' | 'stats';

const tabs: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: 'transcript', label: 'Transcript', icon: <FileText size={18} /> },
  { id: 'correction', label: 'Correction', icon: <Sparkles size={18} /> },
  { id: 'emotion', label: 'Emotion', icon: <Smile size={18} /> },
  { id: 'questions', label: 'Questions', icon: <HelpCircle size={18} /> },
  { id: 'stats', label: 'Stats', icon: <BarChart3 size={18} /> },
];

const emotionColors: Record<string, string> = {
  angry: '#EF4444',
  calm: '#8B5CF6',
  disgust: '#F59E0B',
  fear: '#F97316',
  happy: '#10B981',
  neutral: '#94A3B8',
  sad: '#3B82F6',
  surprised: '#EC4899',
};

const emotionIcons: Record<string, string> = {
  angry: '😠', calm: '😌', disgust: '🤢', fear: '😨',
  happy: '😊', neutral: '😐', sad: '😢', surprised: '😲',
};

export default function ResultsPage({ result, onNavigate }: ResultsPageProps) {
  const [activeTab, setActiveTab] = useState<TabId>('transcript');
  const [copied, setCopied] = useState(false);
  const [expandedSegment, setExpandedSegment] = useState<number | null>(null);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const { whisper, correction, emotion, questions, timing, duration_sec, filename } = result;

  return (
    <div className="min-h-[100dvh] pt-[72px] bg-auris-bg pb-16">
      {/* Header */}
      <div className="max-w-6xl mx-auto px-6 pt-8">
        <button
          onClick={() => onNavigate('upload')}
          className="text-auris-text-secondary hover:text-auris-text text-sm mb-6 flex items-center gap-1 transition-colors"
        >
          <ArrowLeft size={16} /> Analyze Another Presentation
        </button>

        {/* File info */}
        <div className="flex flex-wrap items-center gap-4 mb-8">
          <div className="card-surface px-4 py-2 flex items-center gap-2">
            <Mic size={16} className="text-auris-blue" />
            <span className="text-auris-text text-sm font-medium">{filename}</span>
          </div>
          <div className="card-surface px-4 py-2 flex items-center gap-2">
            <Clock size={16} className="text-auris-teal" />
            <span className="text-auris-text-secondary text-sm">{duration_sec}s</span>
          </div>
          <div className="card-surface px-4 py-2 flex items-center gap-2">
            <Languages size={16} className="text-auris-amber" />
            <span className="text-auris-text-secondary text-sm">{whisper.word_count} words</span>
          </div>
          <div className="card-surface px-4 py-2 flex items-center gap-2">
            <Zap size={16} className="text-auris-blue" />
            <span className="text-auris-text-secondary text-sm">{timing.total_ms}ms total</span>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex flex-wrap gap-2 mb-8 border-b border-auris-border pb-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-medium transition-all duration-200 border-b-2 ${
                activeTab === tab.id
                  ? 'text-auris-blue border-auris-blue bg-[rgba(59,130,246,0.05)]'
                  : 'text-auris-text-secondary border-transparent hover:text-auris-text hover:bg-auris-card'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="min-h-[400px]">
          {activeTab === 'transcript' && (
            <TranscriptTab
              whisper={whisper}
              expandedSegment={expandedSegment}
              setExpandedSegment={setExpandedSegment}
              onCopy={handleCopy}
              copied={copied}
            />
          )}
          {activeTab === 'correction' && (
            <CorrectionTab correction={correction} onCopy={handleCopy} copied={copied} />
          )}
          {activeTab === 'emotion' && (
            <EmotionTab emotion={emotion} />
          )}
          {activeTab === 'questions' && (
            <QuestionsTab questions={questions} />
          )}
          {activeTab === 'stats' && (
            <StatsTab timing={timing} whisper={whisper} correction={correction} emotion={emotion} duration={duration_sec} />
          )}
        </div>
      </div>
    </div>
  );
}

/* Transcript Tab */
function TranscriptTab({
  whisper,
  expandedSegment,
  setExpandedSegment,
  onCopy,
  copied,
}: {
  whisper: AnalysisResult['whisper'];
  expandedSegment: number | null;
  setExpandedSegment: (i: number | null) => void;
  onCopy: (t: string) => void;
  copied: boolean;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-h3 text-auris-text">Whisper Transcript</h3>
        <button
          onClick={() => onCopy(whisper.text)}
          className="flex items-center gap-1.5 text-sm text-auris-text-secondary hover:text-auris-blue transition-colors"
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      <div className="card-surface p-6 mb-6">
        <p className="text-auris-text leading-relaxed">{whisper.text}</p>
      </div>

      <h4 className="text-sm font-semibold text-auris-text-secondary mb-3">
        Segments ({whisper.segments.length})
      </h4>
      <div className="space-y-2">
        {whisper.segments.map((seg, i) => (
          <div
            key={i}
            className="card-surface p-4 cursor-pointer transition-all hover:bg-auris-card-hover"
            onClick={() => setExpandedSegment(expandedSegment === i ? null : i)}
          >
            <div className="flex items-center gap-3">
              <span className="text-caption text-auris-blue font-mono">
                {seg.start.toFixed(1)}s
              </span>
              <span className="text-auris-text text-sm flex-1 truncate">{seg.text}</span>
              <ChevronDown
                size={16}
                className={`text-auris-text-tertiary transition-transform ${expandedSegment === i ? 'rotate-180' : ''}`}
              />
            </div>
            {expandedSegment === i && (
              <div className="mt-3 pt-3 border-t border-auris-border">
                <p className="text-auris-text-secondary text-sm">{seg.text}</p>
                <div className="flex gap-4 mt-2">
                  <span className="text-caption text-auris-text-tertiary">Start: {seg.start}s</span>
                  <span className="text-caption text-auris-text-tertiary">End: {seg.end}s</span>
                  <span className="text-caption text-auris-text-tertiary">Duration: {(seg.end - seg.start).toFixed(1)}s</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* Correction Tab */
function CorrectionTab({
  correction,
  onCopy,
  copied,
}: {
  correction: AnalysisResult['correction'];
  onCopy: (t: string) => void;
  copied: boolean;
}) {
  if (!correction.enabled) {
    return (
      <div className="text-center py-16">
        <p className="text-auris-text-secondary">Grammar correction is disabled.</p>
      </div>
    );
  }

  const stats = correction.critique_stats;
  const improvementPct = stats.total > 0 ? Math.round((stats.corrected / stats.total) * 100) : 0;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-h3 text-auris-text">Grammar Correction</h3>
        <button
          onClick={() => onCopy(correction.corrected)}
          className="flex items-center gap-1.5 text-sm text-auris-text-secondary hover:text-auris-blue transition-colors"
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="card-surface p-4 text-center">
          <p className="text-2xl font-mono text-auris-amber">{stats.corrected}</p>
          <p className="text-caption text-auris-text-tertiary">Corrected</p>
        </div>
        <div className="card-surface p-4 text-center">
          <p className="text-2xl font-mono text-auris-teal">{stats.kept}</p>
          <p className="text-caption text-auris-text-tertiary">Kept</p>
        </div>
        <div className="card-surface p-4 text-center">
          <p className="text-2xl font-mono text-auris-blue">{improvementPct}%</p>
          <p className="text-caption text-auris-text-tertiary">Improvement</p>
        </div>
      </div>

      <div className="card-surface p-6">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles size={18} className="text-auris-amber" />
          <span className="text-sm font-medium text-auris-amber">AI-Enhanced</span>
          <span className="text-caption text-auris-text-tertiary ml-2">{correction.model}</span>
        </div>
        <p className="text-auris-text leading-relaxed">{correction.corrected}</p>
      </div>
    </div>
  );
}

/* Emotion Tab */
function EmotionTab({ emotion }: { emotion: AnalysisResult['emotion'] }) {
  if (!emotion.enabled) {
    return (
      <div className="text-center py-16">
        <p className="text-auris-text-secondary">Emotion detection is disabled.</p>
      </div>
    );
  }

  const dominantColor = emotionColors[emotion.emotion] || '#94A3B8';
  const sortedProbs = Object.entries(emotion.all_probs)
    .sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <h3 className="text-h3 text-auris-text mb-6">Emotion Detection</h3>

      {/* Dominant Emotion */}
      <div className="card-surface p-8 mb-6 text-center">
        <div
          className="w-20 h-20 rounded-full mx-auto mb-4 flex items-center justify-center text-3xl"
          style={{ background: `${dominantColor}20`, border: `2px solid ${dominantColor}40` }}
        >
          {emotionIcons[emotion.emotion] || '😐'}
        </div>
        <p className="text-2xl font-semibold capitalize mb-1" style={{ color: dominantColor }}>
          {emotion.emotion}
        </p>
        <p className="text-auris-text-secondary text-sm mb-2">
          Confidence: <span className="font-mono text-auris-text">{(emotion.confidence * 100).toFixed(1)}%</span>
        </p>
        <div className="flex items-center justify-center gap-2">
          <span
            className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium ${
              emotion.is_reliable ? 'bg-green-500/10 text-green-400' : 'bg-yellow-500/10 text-yellow-400'
            }`}
          >
            <Activity size={12} />
            {emotion.is_reliable ? 'Reliable' : 'Low Confidence'}
          </span>
        </div>
      </div>

      {/* All Probabilities */}
      <h4 className="text-sm font-semibold text-auris-text-secondary mb-3">All Emotions</h4>
      <div className="space-y-3">
        {sortedProbs.map(([label, prob]) => {
          const pct = (prob * 100).toFixed(1);
          const color = emotionColors[label] || '#94A3B8';
          return (
            <div key={label} className="flex items-center gap-3">
              <span className="w-20 text-sm text-auris-text-secondary capitalize text-right">{label}</span>
              <div className="flex-1 h-6 bg-auris-card rounded-full overflow-hidden relative">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, background: color, opacity: 0.7 }}
                />
                <span className="absolute inset-0 flex items-center px-3 text-xs font-mono text-auris-text">
                  {pct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="card-surface p-4 text-center">
          <p className="text-xl font-mono text-auris-teal">{emotion.realtime_factor}x</p>
          <p className="text-caption text-auris-text-tertiary">Realtime Factor</p>
        </div>
        <div className="card-surface p-4 text-center">
          <p className="text-xl font-mono text-auris-blue">{emotion.inference_ms}ms</p>
          <p className="text-caption text-auris-text-tertiary">Inference Time</p>
        </div>
      </div>
    </div>
  );
}

/* Questions Tab */
function QuestionsTab({ questions }: { questions: AnalysisResult['questions'] }) {
  if (!questions.enabled || questions.error) {
    return (
      <div className="text-center py-16">
        <p className="text-auris-text-secondary">
          {questions.error || 'Question generation is unavailable.'}
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-h3 text-auris-text">Jury Questions</h3>
        <span className="text-caption text-auris-text-tertiary">
          Generated in {questions.latency_ms}ms
        </span>
      </div>

      <div className="space-y-4">
        {questions.questions.map((q, i) => (
          <div key={i} className="card-surface p-5 flex gap-4">
            <div className="w-8 h-8 rounded-full bg-auris-blue/10 flex items-center justify-center flex-shrink-0 mt-0.5">
              <span className="text-sm font-bold text-auris-blue">{i + 1}</span>
            </div>
            <p className="text-auris-text leading-relaxed">{q}</p>
          </div>
        ))}
      </div>

      {questions.questions.length === 0 && (
        <div className="text-center py-16">
          <p className="text-auris-text-secondary">No questions generated.</p>
        </div>
      )}
    </div>
  );
}

/* Stats Tab */
function StatsTab({
  timing,
  whisper,
  correction,
  emotion,
  duration,
}: {
  timing: AnalysisResult['timing'];
  whisper: AnalysisResult['whisper'];
  correction: AnalysisResult['correction'];
  emotion: AnalysisResult['emotion'];
  duration: number;
}) {
  const pipelineStages = [
    { name: 'Extract',      time: timing.extract_ms, color: '#3B82F6' },
    { name: 'Whisper',      time: timing.stage_ms?.whisper ?? Math.round(timing.pipeline_ms * 0.4), color: '#14B8A6' },
    { name: 'Flan-T5',      time: correction.enabled ? (timing.stage_ms?.flan ?? Math.round(timing.pipeline_ms * 0.25)) : 0, color: '#F59E0B' },
    { name: 'Audio Emotion',time: emotion.enabled ? (timing.stage_ms?.emotion ?? emotion.latency_ms) : 0, color: '#8B5CF6' },
    { name: 'Text Emotion', time: emotion.enabled ? (timing.stage_ms?.text_emotion ?? 0) : 0, color: '#A78BFA' },
    { name: 'QG',           time: timing.stage_ms?.qg ?? (correction.enabled ? Math.round(timing.pipeline_ms * 0.15) : 0), color: '#EC4899' },
  ];

  return (
    <div>
      <h3 className="text-h3 text-auris-text mb-6">Pipeline Statistics</h3>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card-surface p-5 text-center">
          <p className="text-3xl font-mono text-auris-blue">{duration.toFixed(1)}s</p>
          <p className="text-caption text-auris-text-tertiary mt-1">Duration</p>
        </div>
        <div className="card-surface p-5 text-center">
          <p className="text-3xl font-mono text-auris-teal">{whisper.word_count}</p>
          <p className="text-caption text-auris-text-tertiary mt-1">Words</p>
        </div>
        <div className="card-surface p-5 text-center">
          <p className="text-3xl font-mono text-auris-amber">{whisper.segments.length}</p>
          <p className="text-caption text-auris-text-tertiary mt-1">Segments</p>
        </div>
        <div className="card-surface p-5 text-center">
          <p className="text-3xl font-mono text-[#EC4899]">{timing.total_ms}ms</p>
          <p className="text-caption text-auris-text-tertiary mt-1">Total Time</p>
        </div>
      </div>

      {/* Pipeline Breakdown */}
      <h4 className="text-sm font-semibold text-auris-text-secondary mb-4">Pipeline Breakdown</h4>
      <div className="card-surface p-6">
        <div className="space-y-4">
          {pipelineStages.map((stage) => (
            <div key={stage.name} className="flex items-center gap-4">
              <span className="w-20 text-sm text-auris-text-secondary text-right">{stage.name}</span>
              <div className="flex-1 h-8 bg-auris-bg rounded-lg overflow-hidden relative">
                {stage.time > 0 && (
                  <div
                    className="h-full rounded-lg transition-all duration-500 flex items-center px-3"
                    style={{
                      width: `${Math.min(100, (stage.time / timing.total_ms) * 100)}%`,
                      background: `${stage.color}30`,
                      borderLeft: `3px solid ${stage.color}`,
                    }}
                  >
                    <span className="text-xs font-mono" style={{ color: stage.color }}>
                      {stage.time}ms
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
        <div className="card-surface p-4">
          <p className="text-caption text-auris-text-tertiary mb-1">Transcription</p>
          <p className="text-sm text-auris-text font-medium">Whisper base.en</p>
        </div>
        <div className="card-surface p-4">
          <p className="text-caption text-auris-text-tertiary mb-1">Grammar</p>
          <p className="text-sm text-auris-text font-medium">{correction.model || 'Flan-T5'}</p>
        </div>
        <div className="card-surface p-4">
          <p className="text-caption text-auris-text-tertiary mb-1">Emotion</p>
          <p className="text-sm text-auris-text font-medium">Wav2Vec2 (8 classes)</p>
        </div>
      </div>
    </div>
  );
}
