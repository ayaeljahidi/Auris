import { useState } from 'react'
import { motion } from 'framer-motion'

export default function VadTimeline({ vad_segments, duration_sec, darkMode }) {
  const [tooltip, setTooltip] = useState(null)

  const fmt = (s) => {
    const m = Math.floor(s / 60)
    const sec = (s % 60).toFixed(1)
    return `${m}:${sec.padStart(4, '0')}`
  }

  const totalDuration = duration_sec || (vad_segments?.length > 0
    ? vad_segments[vad_segments.length - 1].end + 1
    : 60)

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className={`rounded-2xl p-6 border-l-4 ${darkMode ? 'bg-gray-900' : 'bg-white shadow-lg'}`}
      style={{ borderLeftColor: '#F59E0B' }}
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'rgba(30,64,175,0.12)', color: '#1E40AF' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </div>
        <div>
          <h3 className="font-bold text-sm" style={{ fontFamily: 'Syne, sans-serif' }}>Voice Activity Timeline</h3>
          <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            {vad_segments?.length || 0} speech segments · {fmt(totalDuration)} total
          </p>
        </div>
      </div>

      {/* Timeline bar */}
      <div className="relative">
        <div className={`h-10 rounded-xl w-full relative overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
          {vad_segments?.map((seg, i) => {
            const left = (seg.start / totalDuration) * 100
            const width = ((seg.end - seg.start) / totalDuration) * 100
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, scaleY: 0 }}
                animate={{ opacity: 1, scaleY: 1 }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
                className="absolute top-0 bottom-0 rounded-lg cursor-pointer"
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 0.5)}%`,
                  background: `rgba(30,64,175,${0.4 + seg.confidence * 0.6})`,
                }}
                onMouseEnter={(e) => setTooltip({ seg, x: e.clientX })}
                onMouseLeave={() => setTooltip(null)}
              />
            )
          })}
        </div>

        {/* Time labels */}
        <div className="flex justify-between mt-2">
          <span className={`text-xs font-mono ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>0:00</span>
          <span className={`text-xs font-mono ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{fmt(totalDuration / 2)}</span>
          <span className={`text-xs font-mono ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{fmt(totalDuration)}</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-4">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm" style={{ background: 'rgba(30,64,175,0.7)' }}/>
          <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Speech</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={`w-3 h-3 rounded-sm ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}/>
          <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Silence</span>
        </div>
        <span className={`text-xs ml-auto ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          Opacity = confidence
        </span>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div className="fixed z-50 pointer-events-none px-3 py-2 rounded-lg text-xs font-mono text-white shadow-lg"
          style={{ left: tooltip.x + 12, top: 100, background: '#1E40AF' }}>
          {fmt(tooltip.seg.start)} → {fmt(tooltip.seg.end)}<br/>
          Confidence: {(tooltip.seg.confidence * 100).toFixed(0)}%
        </div>
      )}
    </motion.div>
  )
}