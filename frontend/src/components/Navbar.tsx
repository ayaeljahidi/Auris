import { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import type { PageView } from '@/types';

interface NavbarProps {
  currentView: PageView;
  onNavigate: (view: PageView) => void;
}

export default function Navbar({ currentView, onNavigate }: NavbarProps) {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isHome = currentView === 'home';

  const scrollToSection = (id: string) => {
    if (!isHome) {
      onNavigate('home');
      setTimeout(() => {
        document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    }
    setMobileOpen(false);
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 h-[72px] flex items-center transition-all duration-300 ${
        scrolled
          ? 'bg-[rgba(10,15,26,0.85)] backdrop-blur-[16px] border-b border-[rgba(148,163,184,0.06)]'
          : 'bg-transparent'
      }`}
    >
      <div className="w-full max-w-7xl mx-auto px-6 flex items-center justify-between">
        {/* Logo */}
        <button
          onClick={() => onNavigate('home')}
          className="flex items-center gap-2.5 group"
        >
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" className="transition-transform duration-200 group-hover:scale-105">
            <path
              d="M14 2L4 24H8L10.5 18H17.5L20 24H24L14 2ZM12 14L14 8L16 14H12Z"
              fill="#3B82F6"
            />
            <path
              d="M14 2L4 24H8L10.5 18H17.5L20 24H24L14 2Z"
              stroke="#3B82F6"
              strokeWidth="1.5"
              fill="none"
            />
            <path d="M6 20L22 20" stroke="#3B82F6" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M7 17L21 17" stroke="#3B82F6" strokeWidth="1" strokeLinecap="round" opacity="0.5" />
          </svg>
          <span className="text-[#F1F5F9] font-bold text-[0.9375rem]">Auris</span>
        </button>

        {/* Center Nav Links - Desktop */}
        <div className="hidden md:flex items-center gap-8">
          {[
            { label: 'About', id: 'about' },
            { label: 'How It Works', id: 'process' },
            { label: 'Features', id: 'features' },
            { label: 'FAQ', id: 'faq' },
          ].map((item) => (
            <button
              key={item.id}
              onClick={() => scrollToSection(item.id)}
              className="text-[0.9375rem] font-semibold text-[#94A3B8] hover:text-[#F1F5F9] transition-all duration-200 hover:-translate-y-px"
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Right CTAs */}
        <div className="hidden md:flex items-center gap-3">
          <button
            onClick={() => onNavigate('upload')}
            className="px-6 py-2.5 rounded-full text-[0.9375rem] font-semibold text-[#F1F5F9] border border-[rgba(148,163,184,0.15)] bg-transparent hover:bg-auris-card hover:border-auris-blue transition-all duration-200"
          >
            Upload Audio
          </button>
          <button
            onClick={() => onNavigate('live')}
            className="px-7 py-2.5 rounded-full text-[0.9375rem] font-semibold text-white bg-auris-blue hover:bg-auris-blue-hover shadow-glow-blue hover:shadow-glow-blue-hover transition-all duration-200 hover:-translate-y-px"
          >
            Analyze
          </button>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="md:hidden text-[#F1F5F9] p-2"
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="absolute top-[72px] left-0 right-0 bg-[rgba(10,15,26,0.95)] backdrop-blur-[16px] border-b border-[rgba(148,163,184,0.08)] md:hidden">
          <div className="px-6 py-6 flex flex-col gap-4">
            {[
              { label: 'About', id: 'about' },
              { label: 'How It Works', id: 'process' },
              { label: 'Features', id: 'features' },
              { label: 'FAQ', id: 'faq' },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => scrollToSection(item.id)}
                className="text-left text-[#94A3B8] hover:text-[#F1F5F9] font-semibold py-2 transition-colors"
              >
                {item.label}
              </button>
            ))}
            <div className="flex flex-col gap-3 pt-4 border-t border-[rgba(148,163,184,0.08)]">
              <button
                onClick={() => { onNavigate('upload'); setMobileOpen(false); }}
                className="w-full py-3 rounded-full text-[#F1F5F9] font-semibold border border-[rgba(148,163,184,0.15)] bg-transparent"
              >
                Upload Audio
              </button>
              <button
                onClick={() => { onNavigate('live'); setMobileOpen(false); }}
                className="w-full py-3 rounded-full text-white font-semibold bg-auris-blue"
              >
                Analyze
              </button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
