import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const steps = [
  { num: '01', title: 'Upload', desc: 'Drop your video or audio file — any format supported. MP4, MOV, AVI, MP3, WAV, M4A and more.' },
  { num: '02', title: 'Voice Analysis', desc: 'Silero VAD isolates speech from silence with precision. Whisper transcribes every word with timestamps.' },
  { num: '03', title: 'Transcription', desc: 'Whisper converts your speech to accurate timestamped text with word-level precision.' },
  { num: '04', title: 'Correction', desc: 'Flan-T5 fixes grammar and improves natural flow. Wav2Vec2 detects emotional tone. QG generates jury-style questions.' },
];

export default function ProcessSteps() {
  const sectionRef = useRef<HTMLElement>(null);
  const stepsRef = useRef<(HTMLDivElement | null)[]>([]);
  const lineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Steps stagger in
      stepsRef.current.forEach((step, i) => {
        if (!step) return;
        gsap.fromTo(
          step,
          { opacity: 0, x: -20 },
          {
            opacity: 1,
            x: 0,
            duration: 0.6,
            ease: 'power2.out',
            delay: i * 0.15,
            scrollTrigger: { trigger: sectionRef.current, start: 'top 75%' },
          }
        );
      });

      // Traveling dot on line
      if (lineRef.current) {
        const dot = lineRef.current.querySelector('.traveling-dot');
        if (dot) {
          gsap.fromTo(
            dot,
            { left: '0%' },
            {
              left: '100%',
              duration: 2,
              ease: 'power2.inOut',
              scrollTrigger: { trigger: sectionRef.current, start: 'top 70%' },
            }
          );
        }
      }
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} id="process" className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg-secondary">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <span className="label-tag text-auris-blue block mb-3">HOW IT WORKS</span>
          <h2 className="text-h2 text-auris-text">From upload to insights in seconds</h2>
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative">
          {steps.map((step, i) => (
            <div
              key={step.num}
              ref={(el) => { stepsRef.current[i] = el; }}
              className="relative"
            >
              {/* Step number background */}
              <span className="absolute -top-4 -left-2 font-mono text-[clamp(2.5rem,5vw,4rem)] text-[rgba(59,130,246,0.15)] leading-none select-none pointer-events-none">
                {step.num}
              </span>
              <div className="relative z-10 pt-6">
                <span className="text-caption text-auris-blue mb-3 block">{step.num}</span>
                <h3 className="text-h3 text-auris-text mb-2">{step.title}</h3>
                <p className="text-body-sm text-auris-text-secondary">{step.desc}</p>
              </div>
            </div>
          ))}

          {/* Connecting line - desktop only */}
          <div
            ref={lineRef}
            className="hidden lg:block absolute top-16 left-[12.5%] right-[12.5%] h-0"
            style={{ borderTop: '2px dotted rgba(148,163,184,0.08)' }}
          >
            <div
              className="traveling-dot absolute -top-1 w-1 h-1 rounded-full bg-auris-blue shadow-[0_0_6px_#3B82F6]"
              style={{ left: '0%' }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
