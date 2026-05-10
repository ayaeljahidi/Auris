import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import HeroTerrain from '@/components/HeroTerrain';
import type { PageView } from '@/types';

interface HeroProps {
  onNavigate: (view: PageView) => void;
}

export default function Hero({ onNavigate }: HeroProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ delay: 0.3 });
      tl.fromTo('.hero-eyebrow', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' })
        .fromTo('.hero-headline-1', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, '-=0.3')
        .fromTo('.hero-headline-2', { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, '-=0.3')
        .fromTo('.hero-sub', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, '-=0.3')
        .fromTo('.hero-cta', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6, ease: 'power3.out' }, '-=0.3')
        .fromTo('.hero-stats', { opacity: 0, y: 15 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out' }, '-=0.2');
    }, contentRef);
    return () => ctx.revert();
  }, []);

  return (
    <section className="relative w-full min-h-[100dvh] flex flex-col items-center justify-center overflow-hidden">
      <HeroTerrain />
      <div ref={contentRef} className="relative z-10 flex flex-col items-center text-center px-6 max-w-[900px] -mt-10">
        {/* Eyebrow */}
        <div className="hero-eyebrow opacity-0 flex items-center gap-2 mb-6">
          <span className="w-2 h-2 rounded-full bg-auris-teal shadow-[0_0_8px_#14B8A6] animate-pulse-dot" />
          <span className="label-tag text-auris-blue">AI-POWERED SPEECH ANALYSIS</span>
        </div>

        {/* Headline */}
        <h1 className="mb-4">
          <span className="hero-headline-1 opacity-0 block font-display text-display text-auris-text">
            Your AI
          </span>
          <span className="hero-headline-2 opacity-0 block font-display text-display text-gradient mt-1">
            Presentation Coach
          </span>
        </h1>

        {/* Subheadline */}
        <p className="hero-sub opacity-0 text-body-lg text-auris-text-secondary max-w-[620px] mb-8">
          Upload your presentation video or audio file and get instant AI-powered feedback on your speech clarity, delivery, emotional tone, and speaking confidence.
        </p>

        {/* CTA Row */}
        <div className="hero-cta opacity-0 flex flex-wrap items-center justify-center gap-4 mb-12">
          <button
            onClick={() => onNavigate('upload')}
            className="btn-primary flex items-center gap-2 group"
          >
            Analyze My Presentation
            <span className="transition-transform duration-200 group-hover:translate-x-1">→</span>
          </button>
          <button
            onClick={() => {
              document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="btn-secondary"
          >
            Learn more ↓
          </button>
        </div>

        {/* Stats */}
        <div className="hero-stats opacity-0 flex flex-wrap items-center justify-center gap-3">
          {[
            { label: '3 AI Models', color: 'blue' },
            { label: '<30s Processing', color: 'teal' },
            { label: '100% Private', color: 'amber' },
          ].map((stat) => (
            <span
              key={stat.label}
              className={`px-4 py-1.5 rounded-full text-caption font-medium ${
                stat.color === 'blue'
                  ? 'bg-[rgba(59,130,246,0.1)] border border-[rgba(59,130,246,0.2)] text-auris-blue'
                  : stat.color === 'teal'
                  ? 'bg-[rgba(20,184,166,0.1)] border border-[rgba(20,184,166,0.2)] text-auris-teal'
                  : 'bg-[rgba(245,158,11,0.1)] border border-[rgba(245,158,11,0.2)] text-auris-amber'
              }`}
            >
              {stat.label}
            </span>
          ))}
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10">
        <div className="w-px h-10 bg-auris-text-tertiary/30 relative overflow-hidden">
          <div className="w-full h-3 bg-auris-text-tertiary animate-scroll-line" />
        </div>
      </div>
    </section>
  );
}
