import { motion } from 'framer-motion'

const pills = [
  { label: 'FFmpeg',        key: 'ffmpeg_ms' },
  { label: 'VAD',           key: 'vad_ms' },
  { label: 'Whisper+Flan',  key: 'parallel_ms' },
  { label: 'Correction',    key: 'correction_ms' },
  { label: 'Total',         key: 'total_ms' },
]

export default function TimingBar({ timing, darkMode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className="flex flex-wrap gap-3 justify-center"
    >
      {pills.map((p, i) => (
        <motion.div
          key={p.key}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.08 }}
          className="flex items-center gap-2 px-4 py-2 rounded-xl"
          style={{ background: p.key === 'total_ms' ? '#1E40AF' : darkMode ? 'rgba(30,64,175,0.15)' : 'rgba(30,64,175,0.08)' }}
        >
          <span className={`text-xs font-medium ${
            p.key === 'total_ms'
              ? 'text-white'
              : darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>{p.label}</span>
          <span className="text-xs font-bold font-mono" style={{ color: '#F59E0B' }}>
            {timing?.[p.key] != null ? `${timing[p.key]}ms` : '—'}
          </span>
        </motion.div>
      ))}
    </motion.div>
  )
}