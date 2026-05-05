import { useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import FeatureCards from '../components/FeatureCards'

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
}

const qaData = [
  {
    q: 'What file formats does Auris support?',
    a: 'Auris supports all major video and audio formats including MP4, MOV, AVI, MKV, MP3, WAV, M4A and more — powered by FFmpeg under the hood.',
  },
  {
    q: 'How long does the analysis take?',
    a: 'Most presentations are processed in under 30 seconds. The pipeline runs Silero VAD, Whisper transcription, and Flan-T5 grammar correction in an optimized sequence.',
  },
  {
    q: 'Is my data private?',
    a: 'Yes. Everything runs locally on your machine. Your files are never sent to any external server — the backend is a local FastAPI instance you control.',
  },
  {
    q: 'Can I use it for languages other than English?',
    a: "Currently optimized for English using the Whisper base.en model. Multi-language support can be enabled by switching the WHISPER_MODEL environment variable.",
  },
  {
    q: 'What is the grammar correction powered by?',
    a: "The correction layer uses Google's Flan-T5 model, a powerful sequence-to-sequence transformer that rewrites your transcript with proper grammar and natural flow.",
  },
]

function QAItem({ item, darkMode }) {
  const [open, setOpen] = useState(false)
  return (
    <motion.div
      variants={fadeUp}
      className={`rounded-2xl border overflow-hidden transition-all duration-300 ${
        darkMode
          ? 'border-white/5 bg-gray-900 hover:border-blue-700/40'
          : 'border-gray-200 bg-white hover:border-blue-300 shadow-sm'
      }`}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-6 text-left"
      >
        <span className={`font-semibold text-sm pr-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}
          style={{ fontFamily: 'Syne, sans-serif' }}>
          {item.q}
        </span>
        <motion.span
          animate={{ rotate: open ? 45 : 0 }}
          transition={{ duration: 0.2 }}
          className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-lg font-light text-white"
          style={{ background: '#1E40AF' }}
        >
          +
        </motion.span>
      </button>
      <motion.div
        initial={false}
        animate={{ height: open ? 'auto' : 0, opacity: open ? 1 : 0 }}
        transition={{ duration: 0.3 }}
        className="overflow-hidden"
      >
        <p className={`px-6 pb-6 text-sm leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {item.a}
        </p>
      </motion.div>
    </motion.div>
  )
}

function HeroAnimation({ darkMode }) {
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      {[0, 1, 2].map(i => (
        <motion.div
          key={i}
          className="absolute rounded-full border"
          style={{
            width: 200 + i * 90,
            height: 200 + i * 90,
            borderColor: i === 0
              ? 'rgba(30,64,175,0.5)'
              : i === 1
              ? 'rgba(30,64,175,0.25)'
              : 'rgba(30,64,175,0.1)',
          }}
          animate={{ scale: [1, 1.04, 1], opacity: [0.7, 1, 0.7] }}
          transition={{ duration: 2.5 + i * 0.5, repeat: Infinity, delay: i * 0.4 }}
        />
      ))}

      {/* Center card */}
      <motion.div
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        className={`relative z-10 w-40 h-40 rounded-3xl flex flex-col items-center justify-center ${
          darkMode ? 'bg-gray-900 border border-white/10' : 'bg-white border border-gray-100'
        }`}
        style={{ boxShadow: '0 0 60px rgba(30,64,175,0.3)' }}
      >
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3"
          style={{ background: '#1E40AF' }}>
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
        </div>
        <span className="text-xs font-bold" style={{ color: '#1E40AF', fontFamily: 'Syne, sans-serif' }}>Analyzing…</span>
      </motion.div>

      {/* Floating chips */}
      <motion.div
        animate={{ y: [0, -8, 0], x: [0, 4, 0] }}
        transition={{ duration: 2.8, repeat: Infinity, delay: 0.5 }}
        className={`absolute top-6 right-6 px-4 py-2 rounded-xl text-xs font-bold shadow-lg border ${
          darkMode ? 'bg-gray-900 border-white/5 text-white' : 'bg-white border-gray-100 text-gray-800 shadow-md'
        }`}
      >
        <span style={{ color: '#F59E0B' }}>✦</span> 186 words detected
      </motion.div>

      <motion.div
        animate={{ y: [0, 7, 0], x: [0, -4, 0] }}
        transition={{ duration: 3.2, repeat: Infinity, delay: 1 }}
        className={`absolute bottom-12 left-4 px-4 py-2 rounded-xl text-xs font-bold shadow-lg border ${
          darkMode ? 'bg-gray-900 border-white/5 text-white' : 'bg-white border-gray-100 text-gray-800 shadow-md'
        }`}
      >
        <span style={{ color: '#1E40AF' }}>●</span> 4 speech segments
      </motion.div>

      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ duration: 2.5, repeat: Infinity, delay: 1.5 }}
        className={`absolute bottom-20 right-2 px-4 py-2 rounded-xl text-xs font-bold shadow-lg border ${
          darkMode ? 'bg-gray-900 border-white/5 text-white' : 'bg-white border-gray-100 text-gray-800 shadow-md'
        }`}
      >
        Grammar ✓
      </motion.div>

      {/* Waveform */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 flex items-end gap-1 pb-2">
        {[4, 7, 12, 9, 14, 8, 11, 6, 13, 7, 5, 10, 8].map((h, i) => (
          <motion.div
            key={i}
            className="w-1.5 rounded-full"
            style={{ height: h * 2, background: i % 3 === 0 ? '#F59E0B' : '#1E40AF', opacity: 0.8 }}
            animate={{ scaleY: [1, 1.6, 0.6, 1.3, 1] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.1, ease: 'easeInOut' }}
          />
        ))}
      </div>
    </div>
  )
}

export default function Landing({ darkMode }) {
  const navigate = useNavigate()

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      {/* ── HERO ── */}
      <section className={`min-h-screen flex items-center relative overflow-hidden pt-16 ${
        darkMode ? 'gradient-hero' : 'gradient-hero-light'
      }`}>
        <div className="absolute top-1/3 right-1/4 w-80 h-80 rounded-full blur-3xl opacity-10 pointer-events-none" style={{ background: '#1E40AF' }} />
        <div className="absolute bottom-1/4 left-1/3 w-48 h-48 rounded-full blur-3xl opacity-8 pointer-events-none" style={{ background: '#F59E0B' }} />

        <div className="max-w-7xl mx-auto px-6 w-full grid grid-cols-1 lg:grid-cols-2 gap-16 items-center py-20">
          {/* LEFT */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-8 border"
              style={{ background: 'rgba(30,64,175,0.1)', borderColor: 'rgba(30,64,175,0.3)', color: '#1E40AF' }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#1E40AF' }} />
              AI-Powered Presentation Coach
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.7 }}
              className={`text-5xl md:text-6xl lg:text-7xl font-black leading-tight mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}
              style={{ fontFamily: 'Syne, sans-serif' }}
            >
              Your AI<br />
              <span style={{ color: '#1E40AF' }}>Presentation</span><br />
              Coach
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45 }}
              className={`text-lg leading-relaxed mb-10 max-w-lg ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}
            >
              Upload your presentation video and get instant AI feedback on your
              speech clarity, delivery rhythm, grammar, and speaking confidence.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="flex items-center gap-4 flex-wrap"
            >
              <motion.button
                whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(30,64,175,0.5)' }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate('/upload')}
                className="px-8 py-4 rounded-2xl text-white font-bold text-base"
                style={{ background: '#1E40AF' }}
              >
                Analyze My Presentation →
              </motion.button>
              <a href="#about"
                className={`text-sm font-medium transition-colors ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-800'}`}>
                Learn more ↓
              </a>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
              className="flex items-center gap-10 mt-12"
            >
              {[
                { num: '3', label: 'AI Models' },
                { num: '<30s', label: 'Processing' },
                { num: '100%', label: 'Private' },
              ].map((s, i) => (
                <div key={i}>
                  <p className="text-2xl font-black" style={{ color: '#1E40AF', fontFamily: 'Syne, sans-serif' }}>{s.num}</p>
                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{s.label}</p>
                </div>
              ))}
            </motion.div>
          </div>

          {/* RIGHT */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="h-96 lg:h-[500px] hidden lg:block"
          >
            <HeroAnimation darkMode={darkMode} />
          </motion.div>
        </div>

        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <div className={`w-6 h-10 rounded-full border-2 flex items-start justify-center p-1.5 ${darkMode ? 'border-white/20' : 'border-blue-300/60'}`}>
            <div className="w-1 h-2 rounded-full" style={{ background: '#1E40AF' }} />
          </div>
        </motion.div>
      </section>

      {/* ── ABOUT ── */}
      <section id="about" className={`py-28 px-6 ${darkMode ? 'bg-gray-950' : 'bg-gray-50'}`}>
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.12 } } }}
            className="space-y-3"
          >
            {[
              { step: '01', label: 'Upload', desc: 'Drop your video or audio file — any format supported' },
              { step: '02', label: 'Voice Analysis', desc: 'Silero VAD isolates speech from silence with precision' },
              { step: '03', label: 'Transcription', desc: 'Whisper converts your speech to accurate timestamped text' },
              { step: '04', label: 'Correction', desc: 'Flan-T5 fixes grammar and improves natural flow' },
            ].map((item, i) => (
              <motion.div
                key={i}
                variants={fadeUp}
                className={`flex items-center gap-4 p-5 rounded-2xl border ${
                  darkMode ? 'bg-gray-900 border-white/5' : 'bg-white border-gray-100 shadow-sm'
                }`}
              >
                <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 text-white text-xs font-bold"
                  style={{ background: i === 3 ? '#F59E0B' : '#1E40AF' }}>
                  {item.step}
                </div>
                <div>
                  <p className="font-bold text-sm" style={{ fontFamily: 'Syne, sans-serif' }}>{item.label}</p>
                  <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{item.desc}</p>
                </div>
                {i < 3 && <div className="ml-auto text-gray-500 text-base">↓</div>}
              </motion.div>
            ))}
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
          >
            <p className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: '#1E40AF' }}>
              About Auris
            </p>
            <h2 className="text-4xl font-black mb-6 leading-tight" style={{ fontFamily: 'Syne, sans-serif' }}>
              Built for students who want to{' '}
              <span style={{ color: '#1E40AF' }}>present with confidence</span>
            </h2>
            <p className={`text-base leading-relaxed mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Auris is an AI coaching tool designed to help students improve not just
              the content of their presentations, but the way they deliver them.
            </p>
            <p className={`text-base leading-relaxed mb-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              By combining speech recognition, voice activity detection, and grammar
              correction into a single pipeline, Auris gives you actionable feedback
              in seconds — completely offline and private.
            </p>
            <motion.button
              whileHover={{ scale: 1.04, boxShadow: '0 0 30px rgba(30,64,175,0.4)' }}
              whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/upload')}
              className="px-7 py-3 rounded-xl text-white text-sm font-semibold"
              style={{ background: '#1E40AF' }}
            >
              Try it now →
            </motion.button>
          </motion.div>
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section id="features" className={`${darkMode ? 'bg-gray-950' : 'bg-white'}`}>
        <FeatureCards darkMode={darkMode} />
      </section>

      {/* ── FAQ ── */}
      <section id="faq" className={`py-28 px-6 ${darkMode ? 'bg-gray-950' : 'bg-gray-50'}`}>
        <div className="max-w-3xl mx-auto">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={fadeUp}
            className="text-center mb-14"
          >
            <p className="text-xs font-semibold tracking-widest uppercase mb-3" style={{ color: '#1E40AF' }}>FAQ</p>
            <h2 className="text-4xl font-black" style={{ fontFamily: 'Syne, sans-serif' }}>
              Frequently asked <span style={{ color: '#1E40AF' }}>questions</span>
            </h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.1 } } }}
            className="space-y-3"
          >
            {qaData.map((item, i) => (
              <QAItem key={i} item={item} darkMode={darkMode} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className={`py-16 px-6 border-t ${darkMode ? 'border-white/5 bg-gray-950' : 'border-gray-100 bg-white'}`}>
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div>
              <p className="font-black text-2xl mb-1" style={{ fontFamily: 'Syne, sans-serif' }}>
                <span style={{ color: '#1E40AF' }}>Au</span><span style={{ color: '#F59E0B' }}>ris</span>
              </p>
              <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>Speak better. Present smarter.</p>
            </div>

            <div className="flex items-center gap-8">
              {[['About', '#about'], ['Features', '#features'], ['FAQ', '#faq']].map(([label, href]) => (
                <a key={label} href={href}
                  className={`text-sm font-medium transition-colors ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-800'}`}>
                  {label}
                </a>
              ))}
            </div>

            <motion.button
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => navigate('/upload')}
              className="px-6 py-2.5 rounded-xl text-white text-sm font-semibold"
              style={{ background: '#1E40AF' }}
            >
              Start Analyzing →
            </motion.button>
          </div>

          <div className={`mt-10 pt-8 border-t text-center text-xs ${
            darkMode ? 'border-white/5 text-gray-600' : 'border-gray-100 text-gray-400'
          }`}>
            © 2026 Auris · Built with Whisper · Silero VAD · Flan-T5 · FastAPI
          </div>
        </div>
      </footer>
    </motion.div>
  )
}