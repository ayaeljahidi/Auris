import { motion } from 'framer-motion'

const features = [
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
        <line x1="12" y1="19" x2="12" y2="23"/>
        <line x1="8" y1="23" x2="16" y2="23"/>
      </svg>
    ),
    title: 'Speech Analysis',
    desc: 'Detect pacing, filler words, clarity and rhythm in your spoken delivery with millisecond precision.',
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="4 7 4 4 20 4 20 7"/>
        <line x1="9" y1="20" x2="15" y2="20"/>
        <line x1="12" y1="4" x2="12" y2="20"/>
        <path d="M5 12h14"/>
      </svg>
    ),
    title: 'Grammar Correction',
    desc: 'AI-powered Flan-T5 model rewrites your transcript with perfect grammar and natural flow.',
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
      </svg>
    ),
    title: 'Delivery Feedback',
    desc: 'Get actionable insights on voice activity, silence patterns, and speaking confidence score.',
  },
]

const containerVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.15 } },
}

const cardVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
}

export default function FeatureCards({ darkMode }) {
  return (
    <section className="py-24 px-6 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
        className="text-center mb-16"
      >
        <p className={`text-sm font-semibold tracking-widest uppercase mb-3 ${darkMode ? 'text-yellow-400' : 'text-yellow-600'}`}
          style={{ color: '#F59E0B' }}>
          What Auris does
        </p>
        <h2 className="text-4xl font-bold" style={{ fontFamily: 'Syne, sans-serif' }}>
          Everything you need to{' '}
          <span style={{ color: '#1E40AF' }}>present better</span>
        </h2>
      </motion.div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="grid grid-cols-1 md:grid-cols-3 gap-6"
      >
        {features.map((f, i) => (
          <motion.div
            key={i}
            variants={cardVariants}
            whileHover={{ scale: 1.03, boxShadow: '0 0 30px rgba(30,64,175,0.25)' }}
            className={`relative rounded-2xl p-8 cursor-default transition-all duration-300 border-t-2 ${
              darkMode
                ? 'bg-gray-900 border-yellow-400'
                : 'bg-white border-yellow-400 shadow-lg'
            }`}
            style={{ borderTopColor: '#F59E0B' }}
          >
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-6"
              style={{ background: 'rgba(30,64,175,0.12)', color: '#1E40AF' }}>
              {f.icon}
            </div>
            <h3 className="text-xl font-bold mb-3" style={{ fontFamily: 'Syne, sans-serif' }}>{f.title}</h3>
            <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{f.desc}</p>
          </motion.div>
        ))}
      </motion.div>
    </section>
  )
}