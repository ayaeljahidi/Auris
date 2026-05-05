import { useState } from 'react'
import { motion } from 'framer-motion'

export default function TranscriptCard({ whisper, darkMode }) {
  const [copied, setCopied] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(whisper.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = (s % 60).toFixed(1)
    return `${m}:${sec.padStart(4, '0')}`
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`rounded-2xl overflow-hidden border-l-4 ${
        darkMode ? 'bg-gray-900 border-yellow-400' : 'bg-white shadow-lg border-yellow-400'
      }`}
      style={{ borderLeftColor: '#F59E0B' }}
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(30,64,175,0.12)', color: '#1E40AF' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              </svg>
            </div>
            <div>
              <h3 className="font-bold text-sm" style={{ fontFamily: 'Syne, sans-serif' }}>Whisper Transcript</h3>
              <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {whisper.word_count} words · {whisper.segments?.length || 0} segments
              </p>
            </div>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={copy}
            className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
              copied
                ? 'text-green-400 bg-green-400/10'
                : darkMode ? 'text-gray-400 hover:text-white bg-white/5 hover:bg-white/10' : 'text-gray-500 hover:text-gray-800 bg-gray-100'
            }`}
          >
            {copied ? '✓ Copied' : 'Copy'}
          </motion.button>
        </div>

        {/* Full text */}
        <p className={`text-sm leading-relaxed mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
          {whisper.text}
        </p>

        {/* Segments toggle */}
        {whisper.segments?.length > 0 && (
          <>
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-xs font-semibold flex items-center gap-1 mb-3"
              style={{ color: '#1E40AF' }}
            >
              <span>{expanded ? '▲' : '▼'}</span>
              {expanded ? 'Hide' : 'Show'} timed segments
            </button>

            {expanded && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="space-y-2 max-h-60 overflow-y-auto pr-1"
              >
                {whisper.segments.map((seg, i) => (
                  <div key={i} className={`flex gap-3 text-xs rounded-lg p-2.5 ${
                    darkMode ? 'bg-gray-800' : 'bg-gray-50'
                  }`}>
                    <span className="font-mono shrink-0" style={{ color: '#F59E0B' }}>
                      {formatTime(seg.start)} → {formatTime(seg.end)}
                    </span>
                    <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{seg.text}</span>
                  </div>
                ))}
              </motion.div>
            )}
          </>
        )}
      </div>
    </motion.div>
  )
}