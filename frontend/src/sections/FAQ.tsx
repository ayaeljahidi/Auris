import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Plus } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

const faqs = [
  {
    q: 'What file formats does Auris support?',
    a: 'Auris supports virtually all common audio and video formats including MP4, MOV, AVI, MP3, WAV, M4A, FLAC, OGG, and WebM. If you have a rare format, try converting to MP3 or WAV first.',
  },
  {
    q: 'How long does the analysis take?',
    a: 'Most presentations under 10 minutes are processed in under 30 seconds. Longer files up to 30 minutes typically complete within 2 minutes. The live recording mode provides feedback instantly as you speak.',
  },
  {
    q: 'Is my data private?',
    a: 'Absolutely. All processing happens locally on your device or on our secure servers. We never store or share your audio files. Your presentations remain completely confidential.',
  },
  {
    q: 'Can I use it for languages other than English?',
    a: "Currently, Auris is optimized for English speech analysis. We are working on adding support for Spanish, French, German, and Mandarin in upcoming releases.",
  },
  {
    q: 'What is the grammar correction powered by?',
    a: "Our grammar correction uses Google's Flan-T5 base model running locally on CPU. It detects and fixes grammatical errors, removes filler words, and improves sentence flow while preserving your original meaning.",
  },
  {
    q: 'How accurate is the emotion detection?',
    a: 'We use the Wav2Vec2 speech emotion classification model with 8 emotion categories (angry, calm, disgust, fear, happy, neutral, sad, surprised). Confidence scores above 50% are considered reliable.',
  },
];

function FAQItem({ item }: { item: typeof faqs[0] }) {
  const [open, setOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  return (
    <div className="border-b border-auris-border">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-6 text-left cursor-pointer group"
        aria-expanded={open}
      >
        <span className="text-body font-semibold text-auris-text pr-4 group-hover:text-auris-blue transition-colors">
          {item.q}
        </span>
        <Plus
          size={20}
          className={`text-auris-text-secondary flex-shrink-0 transition-transform duration-300 ${open ? 'rotate-45' : ''}`}
        />
      </button>
      <div
        ref={contentRef}
        className="overflow-hidden transition-all duration-300 ease-out"
        style={{
          maxHeight: open ? (contentRef.current?.scrollHeight || 200) + 'px' : '0px',
        }}
        aria-hidden={!open}
        role="region"
      >
        <p className="text-body-sm text-auris-text-secondary pb-6">{item.a}</p>
      </div>
    </div>
  );
}

export default function FAQ() {
  const sectionRef = useRef<HTMLElement>(null);
  const itemsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const ctx = gsap.context(() => {
      itemsRef.current.forEach((item, i) => {
        if (!item) return;
        gsap.fromTo(
          item,
          { opacity: 0, y: 20 },
          {
            opacity: 1,
            y: 0,
            duration: 0.5,
            ease: 'power2.out',
            delay: i * 0.08,
            scrollTrigger: { trigger: sectionRef.current, start: 'top 80%' },
          }
        );
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} id="faq" className="w-full py-[clamp(80px,12vh,140px)] px-6 bg-auris-bg-secondary">
      <div className="max-w-[800px] mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <span className="label-tag text-auris-amber block mb-3">FAQ</span>
          <h2 className="text-h2 text-auris-text">Frequently asked questions</h2>
        </div>

        {/* Accordion */}
        {faqs.map((item, i) => (
          <div key={i} ref={(el) => { itemsRef.current[i] = el; }}>
            <FAQItem item={item} />
          </div>
        ))}
      </div>
    </section>
  );
}
