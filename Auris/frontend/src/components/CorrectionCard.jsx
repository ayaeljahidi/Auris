import { useState } from 'react'
import { motion } from 'framer-motion'

function diffWords(original, corrected) {
  const origWords = original.trim().split(/\s+/)
  const corrWords = corrected.trim().split(/\s+/)
  const result = []

  let oi = 0, ci = 0
  while (ci < corrWords.length) {
    if (oi < origWords.length && origWords[oi].replace(/[^a-zA-Z0-9]/g, '').toLowerCase() === corrWords[ci].replace(/[^a-zA-Z0-9]/g, '').toLowerCase()) {
      result.push({ word: corrWords[ci], changed: false })
      oi++; ci++
    } else {
      result.push({ word: corrWords[ci], changed: true })
      ci++
      if (oi < origWords.length) oi++
    }
  }
  return result
}

export default function CorrectionCard({ correction, originalText, darkMode }) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(correction.corrected)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!correction.enabled) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className={`rounded-2xl p-6 border-l-4 ${darkMode ? 'bg-gray-900' : 'bg-white shadow-lg'}`}
        style={{ borderLeftColor: '#F59E0B' }}
      >
        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Grammar correction is disabled.</p>
      </motion.div>
    )
  }

  const diffed = diffWords(originalText, correction.corrected)
  const changedCount = diffed.filter(w => w.changed).length

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className={`rounded-2xl overflow-hidden border-l-4 ${
        darkMode ? 'bg-gray-900' : 'bg-white shadow-lg'
      }`}
      style={{ borderLeftColor: '#F59E0B' }}
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(245,158,11,0.12)', color: '#F59E0B' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
              </svg>
            </div>
            <div>
              <h3 className="font-bold text-sm" style={{ fontFamily: 'Syne, sans-serif' }}>Grammar Correction</h3>
              <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {changedCount} word{changedCount !== 1 ? 's' : ''} improved · {correction.latency_ms}ms
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

        {/* Diff view */}
        <div className={`text-sm leading-relaxed rounded-xl p-4 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
          {diffed.map((w, i) => (
            <span key={i}>
              {w.changed ? (
                <span className="font-semibold rounded px-0.5" style={{ color: '#F59E0B', background: 'rgba(245,158,11,0.1)' }}>
                  {w.word}
                </span>
              ) : (
                <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>{w.word}</span>
              )}
              {i < diffed.length - 1 ? ' ' : ''}
            </span>
          ))}
        </div>

        <p className={`text-xs mt-3 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          <span style={{ color: '#F59E0B' }}>Gold</span> words were corrected by Flan-T5 · Model: {correction.model}
        </p>
      </div>
    </motion.div>
  )
}