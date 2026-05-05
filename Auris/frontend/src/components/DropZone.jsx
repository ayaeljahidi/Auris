import { useState, useRef } from 'react'
import { motion } from 'framer-motion'

export default function DropZone({ onFileSelect, darkMode }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) onFileSelect(f)
  }

  const handleChange = (e) => {
    const f = e.target.files[0]
    if (f) onFileSelect(f)
  }

  return (
    <motion.div
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      whileHover={{ scale: 1.01 }}
      animate={dragging ? { scale: 1.02 } : { scale: 1 }}
      className={`relative rounded-2xl border-2 border-dashed p-16 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
        dragging
          ? 'border-yellow-400 bg-yellow-400/5'
          : darkMode
            ? 'border-blue-700/50 hover:border-blue-500 bg-gray-900/50'
            : 'border-blue-300 hover:border-blue-500 bg-blue-50/50'
      }`}
      style={dragging ? { boxShadow: '0 0 30px rgba(245,158,11,0.3)' } : {}}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,audio/*"
        className="hidden"
        onChange={handleChange}
      />

      {/* Animated upload icon */}
      <motion.div
        animate={dragging ? { scale: 1.2, rotate: 5 } : { scale: 1, rotate: 0 }}
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
        style={{ background: dragging ? 'rgba(245,158,11,0.15)' : 'rgba(30,64,175,0.1)' }}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
          stroke={dragging ? '#F59E0B' : '#1E40AF'} strokeWidth="1.5"
          strokeLinecap="round" strokeLinejoin="round">
          <polyline points="16 16 12 12 8 16"/>
          <line x1="12" y1="12" x2="12" y2="21"/>
          <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
        </svg>
      </motion.div>

      <p className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}
        style={{ fontFamily: 'Syne, sans-serif' }}>
        {dragging ? 'Release to upload' : 'Drop your presentation video or audio here'}
      </p>
      <p className={`text-sm mb-6 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Supports MP4, MOV, AVI, MP3, WAV, M4A and more
      </p>

      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.97 }}
        type="button"
        className="px-6 py-2.5 rounded-xl text-sm font-semibold text-white transition-all"
        style={{ background: '#1E40AF' }}
        onClick={(e) => { e.stopPropagation(); inputRef.current.click() }}
      >
        Browse file
      </motion.button>
    </motion.div>
  )
}