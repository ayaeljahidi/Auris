import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Mic, Sparkles, Star } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

const features = [
  {
    icon: <Mic size={24} className="text-auris-blue" />,
    iconBg: 'bg-[rgba(59,130,246,0.1)]',
    title: 'Speech Analysis',
    description: 'Detect pacing, filler words, clarity, and rhythm in your spoken delivery with millisecond precision. Our AI breaks down every utterance into actionable insights.',
    accent: 'waveform' as const,
  },
  {
    icon: <Sparkles size={24} className="text-auris-amber" />,
    iconBg: 'bg-[rgba(245,158,11,0.1)]',
    title: 'Grammar Correction',
    description: 'AI-powered Flan-T5 model rewrites your transcript with perfect grammar and natural flow. Eliminate filler words, fix sentence structure, and sound polished.',
    accent: 'accuracy' as const,
  },
  {
    icon: <Star size={24} className="text-auris-teal" />,
    iconBg: 'bg-[rgba(20,184,166,0.1)]',
    title: 'Delivery Feedback',
    description: 'Get actionable insights on voice activity, emotional tone, silence patterns, and speaking confidence. Understand how your audience perceives every word.',
    accent: 'confidence' as const,
  },
];

function WaveformBars() {
  return (
    <div className="flex items-end gap-[3px] h-8">
      {[8, 16, 32, 20, 12, 24].map((h, i) => (
        <div
          key={i}
          className="w-[3px] bg-auris-blue rounded-sm animate-wave origin-bottom"
          style={{ height: `${h}px`, animationDelay: `${i * 0.1}s` }}
        />
      ))}
    </div>
  );
}

function RippleIndicator() {
  return (
    <div className="relative w-10 h-10 flex items-center justify-center">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="absolute rounded-full border-2 border-auris-teal animate-ripple"
          style={{
            width: `${40 + i * 20}px`,
            height: `${40 + i * 20}px`,
            animationDelay: `${i * 0.6}s`,
          }}
        />
      ))}
    </div>
  );
}

export default function Features() {
  const sectionRef = useRef<HTMLElement>(null);
  const cardsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Parallax speed
      cardsRef.current.forEach((card, i) => {
        if (!card) return;
        const speed = [0.95, 1.0, 1.05][i];
        gsap.to(card, {
          y: (1 - speed) * 150,
          ease: 'none',
          scrollTrigger: { trigger: card, start: 'top bottom', end: 'bottom top', scrub: true },
        });

        // Bouncy entrance
        gsap.fromTo(
          card,
          { scaleY: 0.95, y: 80, opacity: 0 },
          {
            scaleY: 1,
            y: 0,
            opacity: 1,
            ease: 'elastic.out(0.9, 0.4)',
            scrollTrigger: { trigger: card, start: 'top bottom-=10%', end: 'top top+=40%', scrub: true },
          }
        );
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="features"
      className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg-secondary"
    >
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <span className="label-tag text-auris-amber block mb-3">WHAT AURIS DOES</span>
          <h2 className="text-h2 text-auris-text">
            Everything you need to{' '}
            <span className="text-gradient">present</span> better
          </h2>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, i) => (
            <div
              key={feature.title}
              ref={(el) => { cardsRef.current[i] = el; }}
              data-speed={[0.95, 1.0, 1.05][i]}
              className="card-surface p-10 transition-all duration-300 hover:border-auris-border-active hover:-translate-y-1 hover:shadow-card will-change-transform group"
            >
              <div className={`w-12 h-12 ${feature.iconBg} rounded-xl flex items-center justify-center mb-6`}>
                {feature.icon}
              </div>
              <h3 className="text-h3 text-auris-text mb-3">{feature.title}</h3>
              <p className="text-body-sm text-auris-text-secondary mb-6">{feature.description}</p>

              {feature.accent === 'waveform' && (
                <div className="flex justify-end">
                  <WaveformBars />
                </div>
              )}
              {feature.accent === 'accuracy' && (
                <div className="flex justify-end items-end flex-col">
                  <span className="font-mono text-2xl text-auris-amber">99.2%</span>
                  <span className="text-caption text-auris-text-tertiary">accuracy</span>
                </div>
              )}
              {feature.accent === 'confidence' && (
                <div className="flex justify-end">
                  <RippleIndicator />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
