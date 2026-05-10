import type { PageView } from '@/types';

interface FooterProps {
  onNavigate: (view: PageView) => void;
}

export default function Footer({ onNavigate }: FooterProps) {
  return (
    <footer className="bg-auris-bg border-t border-auris-border">
      <div className="max-w-7xl mx-auto px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-2.5">
          <svg width="24" height="24" viewBox="0 0 28 28" fill="none">
            <path d="M14 2L4 24H8L10.5 18H17.5L20 24H24L14 2Z" stroke="#3B82F6" strokeWidth="1.5" fill="none" />
            <path d="M6 20L22 20" stroke="#3B82F6" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <span className="text-[#F1F5F9] font-bold text-sm">Auris</span>
          <span className="text-auris-text-tertiary text-caption ml-2">Speak better. Present smarter.</span>
        </div>

        <div className="flex items-center gap-6">
          {['About', 'Features', 'FAQ'].map((label) => (
            <button
              key={label}
              onClick={() => onNavigate('home')}
              className="text-sm text-auris-text-secondary hover:text-auris-text transition-colors"
            >
              {label}
            </button>
          ))}
        </div>

        <button
          onClick={() => onNavigate('upload')}
          className="px-6 py-2.5 rounded-full text-sm font-semibold text-white bg-auris-blue hover:bg-auris-blue-hover shadow-glow-blue transition-all duration-200"
        >
          Start Analyzing →
        </button>
      </div>

      <div className="border-t border-auris-border py-5 text-center">
        <p className="text-caption text-auris-text-tertiary">
          © 2025 Auris. Built with Whisper, Silero VAD, Flan-T5, Wav2Vec2 & FastAPI.
        </p>
      </div>
    </footer>
  );
}
