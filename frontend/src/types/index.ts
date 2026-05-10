export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export interface WhisperResult {
  text: string;
  word_count: number;
  segments: TranscriptSegment[];
}

export interface CorrectionResult {
  corrected: string;
  enabled: boolean;
  model: string | null;
  latency_ms: number;
  critique_stats: {
    corrected: number;
    kept: number;
    total: number;
  };
}

export interface EmotionResult {
  enabled: boolean;
  emotion: string;
  confidence: number;
  latency_ms: number;
  is_reliable: boolean;
  all_probs: Record<string, number>;
  realtime_factor: number;
  inference_ms: number;
}

export interface QuestionsResult {
  enabled: boolean;
  questions: string[];
  raw: string;
  latency_ms: number;
  error: string | null;
}

export interface TimingResult {
  extract_ms: number;
  pipeline_ms: number;
  total_ms: number;
  stage_ms?: {
    whisper?: number;
    flan?: number;
    emotion?: number;
    text_emotion?: number;
    qg?: number;
  };
}

export interface AnalysisResult {
  status: string;
  filename: string;
  duration_sec: number;
  whisper: WhisperResult;
  correction: CorrectionResult;
  emotion: EmotionResult;
  questions: QuestionsResult;
  timing: TimingResult;
}

export type PageView = 'home' | 'upload' | 'live' | 'results';
