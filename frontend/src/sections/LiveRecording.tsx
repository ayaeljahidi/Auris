import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Mic, Lock, Zap, Globe } from 'lucide-react';
import type { PageView } from '@/types';

gsap.registerPlugin(ScrollTrigger);

interface LiveRecordingProps {
  onNavigate: (view: PageView) => void;
}

export default function LiveRecording({ onNavigate }: LiveRecordingProps) {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        sectionRef.current,
        { opacity: 0, y: 30 },
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

  return (
    <section
      ref={sectionRef}
      className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg relative overflow-hidden"
      style={{ background: 'radial-gradient(ellipse at center, rgba(59,130,246,0.05) 0%, transparent 70%)' }}
    >
      <div className="max-w-[640px] mx-auto text-center flex flex-col items-center">
        {/* Pulsing circle */}
        <div className="relative w-20 h-20 flex items-center justify-center mb-8">
          {/* Outer rings */}
          <div className="absolute inset-0 rounded-full border-2 border-[rgba(20,184,166,0.3)] animate-ripple" />
          <div className="absolute inset-0 rounded-full border-2 border-[rgba(20,184,166,0.2)] animate-ripple" style={{ animationDelay: '0.6s' }} />
          {/* Inner circle */}
          <div className="w-16 h-16 rounded-full bg-[rgba(20,184,166,0.15)] flex items-center justify-center">
            <Mic size={32} className="text-auris-teal" />
          </div>
        </div>

        <h2 className="text-h1 text-auris-text mb-4">Or record live</h2>
        <p className="text-body-lg text-auris-text-secondary mb-8">
          Start speaking and get real-time feedback as you present. No upload needed — just press record and let AI guide your delivery.
        </p>

        <button
          onClick={() => onNavigate('live')}
          className="px-8 py-3.5 rounded-full font-semibold text-base bg-auris-teal text-auris-bg hover:bg-[#0D9488] hover:shadow-glow-teal transition-all duration-200"
        >
          Start Recording
        </button>

        {/* Status badges */}
        <div className="flex flex-wrap items-center justify-center gap-3 mt-6">
          {[
            { icon: <Lock size={12} />, label: '100% Private' },
            { icon: <Zap size={12} />, label: 'Real-time' },
            { icon: <Globe size={12} />, label: 'Browser-based' },
          ].map((badge) => (
            <span
              key={badge.label}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-full text-caption bg-[rgba(20,184,166,0.08)] text-auris-teal"
            >
              {badge.icon}
              {badge.label}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
