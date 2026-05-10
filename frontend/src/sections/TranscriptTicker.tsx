import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Mic, Sparkles } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

const transcriptPairs = [
  {
    raw: 'So um today we gonna talk about like our new product and it\'s features uh we\'ve been working on this for six months now',
    corrected: 'Today, we are going to discuss our new product and its features. We have been working on this for six months.',
  },
  {
    raw: 'The data shows that user engagement has increased by forty percent which is um significantly higher than what we projected',
    corrected: 'The data shows that user engagement has increased by 40%, which is significantly higher than what we projected.',
  },
  {
    raw: 'I think maybe we should consider um alternative approaches because the current one isn\'t working so well',
    corrected: 'We should consider alternative approaches because the current one is not performing well.',
  },
  {
    raw: 'Our Q3 results were um pretty good actually we exceeded revenue targets by about twelve percent',
    corrected: 'Our Q3 results were strong. We exceeded revenue targets by approximately 12%.',
  },
  {
    raw: 'So yeah the main takeaway here is that we need to um focus more on customer retention going forward',
    corrected: 'The main takeaway is that we need to focus more on customer retention.',
  },
  {
    raw: 'We been working on this feature for a while now and um I think users are really gonna love it',
    corrected: 'We have been working on this feature for some time, and we believe users will love it.',
  },
  {
    raw: 'The feedback from beta testers has been mostly positive although there are um a few issues we still need to address',
    corrected: 'The feedback from beta testers has been mostly positive, although there are a few issues we still need to address.',
  },
  {
    raw: 'Let me just say that this has been a team effort and um I couldn\'t have done it without everyone\'s support',
    corrected: 'This has been a team effort, and it could not have been done without everyone\'s support.',
  },
];

function TickerColumn({
  title,
  icon,
  badge,
  badgeColor,
  transcripts,
  isCorrected,
}: {
  title: string;
  icon: React.ReactNode;
  badge: string;
  badgeColor: string;
  transcripts: string[];
  isCorrected: boolean;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        {icon}
        <span className="text-auris-text font-semibold text-sm">{title}</span>
        <span className={`text-caption px-2.5 py-0.5 rounded-full ${badgeColor}`}>{badge}</span>
      </div>
      <div className="h-[480px] overflow-hidden relative" style={{ maskImage: 'linear-gradient(to bottom, transparent 0%, black 8%, black 92%, transparent 100%)', WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 8%, black 92%, transparent 100%)' }}>
        <div className="flex flex-col animate-ticker" style={{ willChange: 'transform' }}>
          {[...transcripts, ...transcripts].map((text, i) => (
            <div
              key={i}
              className={`mb-6 px-4 py-4 rounded-lg text-body-sm leading-relaxed ${
                isCorrected
                  ? 'bg-[rgba(59,130,246,0.04)] text-auris-text border-l-2 border-[rgba(59,130,246,0.3)]'
                  : 'bg-[rgba(148,163,184,0.04)] text-auris-text-secondary opacity-70'
              }`}
              aria-label={isCorrected ? `After: ${text}` : `Before: ${text}`}
            >
              {text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function TranscriptTicker() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        sectionRef.current,
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: { trigger: sectionRef.current, start: 'top 80%' },
        }
      );
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  const rawTexts = transcriptPairs.map((p) => p.raw);
  const correctedTexts = transcriptPairs.map((p) => p.corrected);

  return (
    <section
      ref={sectionRef}
      className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg relative"
      style={{ background: 'linear-gradient(to bottom, rgba(10,15,26,0) 0%, #0A0F1A 80px)' }}
      aria-label="Grammar correction demo"
    >
      <div className="max-w-[1100px] mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <span className="label-tag text-auris-blue block mb-3">GRAMMAR CORRECTION</span>
          <h2 className="text-h2 text-auris-text mb-4">See the difference</h2>
          <p className="text-body text-auris-text-secondary max-w-[560px] mx-auto">
            Raw speech-to-text output versus AI-enhanced grammar correction. Every transcript is automatically refined for clarity and professionalism.
          </p>
        </div>

        {/* Two Column Ticker */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <TickerColumn
            title="Whisper Transcript"
            icon={<Mic size={16} className="text-auris-text-tertiary" />}
            badge="Raw"
            badgeColor="bg-[rgba(148,163,184,0.08)] text-auris-text-tertiary"
            transcripts={rawTexts}
            isCorrected={false}
          />
          <TickerColumn
            title="Grammar Correction"
            icon={<Sparkles size={16} className="text-auris-amber" />}
            badge="AI-Enhanced"
            badgeColor="bg-[rgba(245,158,11,0.1)] text-auris-amber"
            transcripts={correctedTexts}
            isCorrected={true}
          />
        </div>
      </div>
    </section>
  );
}
