import { useRef } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import useTranscribe from '../hooks/useTranscribe'
import DropZone from '../components/DropZone'
import PipelineSteps from '../components/PipelineSteps'
import TranscriptCard from '../components/TranscriptCard'
import CorrectionCard from '../components/CorrectionCard'
import VadTimeline from '../components/VadTimeline'
import TimingBar from '../components/TimingBar'

export default function Upload({ darkMode }) {
  const navigate = useNavigate()
  const previewRef = useRef()
  const { file, setFile, status, result, error, currentStep, analyze, reset } = useTranscribe()

  const handleFileSelect = (f) => {
    setFile(f)
    if (previewRef.current) {
      previewRef.current.src = URL.createObjectURL(f)
    }
  }

  const isLoading = status === 'loading'
  const isSuccess = status === 'success'

  return (
    <motion.div
      initial={{ opacity: 0, x: 60 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -60 }}
      transition={{ duration: 0.5, ease: 'easeInOut' }}
      className="min-h-screen pt-24 pb-20 px-6"
    >
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <div className="text-center mb-10">
          <motion.button
            onClick={() => navigate('/')}
            className={`text-sm mb-6 flex items-center gap-1 mx-auto transition-colors ${
              darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-400 hover:text-gray-700'
            }`}
            whileHover={{ x: -3 }}
          >
            ← Back to Home
          </motion.button>
          <h1 className="text-4xl font-black mb-2" style={{ fontFamily: 'Syne, sans-serif' }}>
            Analyze Your{' '}
            <span style={{ color: '#1E40AF' }}>Presentation</span>
          </h1>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Upload a video or audio file to get AI-powered speech feedback
          </p>
        </div>

        {/* ── UPLOAD ZONE ── */}
        {!isLoading && !isSuccess && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <DropZone onFileSelect={handleFileSelect} darkMode={darkMode} />

            {/* Preview */}
            {file && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4"
              >
                <div className={`rounded-2xl overflow-hidden ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
                  <video ref={previewRef} controls className="w-full max-h-56 object-contain" />
                </div>
                <div className="flex items-center justify-between mt-3 px-1">
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    📎 {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                  <button onClick={() => setFile(null)} className="text-xs text-red-400 hover:text-red-300">
                    Remove
                  </button>
                </div>
              </motion.div>
            )}

            {/* Analyze button */}
            {file && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                whileHover={{ scale: 1.02, boxShadow: '0 0 30px rgba(30,64,175,0.5)' }}
                whileTap={{ scale: 0.98 }}
                onClick={analyze}
                className="w-full mt-5 py-4 rounded-2xl text-white font-bold text-lg"
                style={{ background: '#1E40AF' }}
              >
                Analyze Presentation ✦
              </motion.button>
            )}

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 p-4 rounded-xl text-sm text-red-400 border border-red-400/20 bg-red-400/5"
              >
                ⚠ {error}
              </motion.div>
            )}
          </motion.div>
        )}

        {/* ── PIPELINE LOADING ── */}
        {isLoading && (
          <PipelineSteps currentStep={currentStep} darkMode={darkMode} />
        )}

        {/* ── RESULTS ── */}
        {isSuccess && result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
          >
            {/* Success badge */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center py-6 mb-6"
            >
              <div className="inline-flex items-center gap-2 px-5 py-2 rounded-full text-sm font-semibold border"
                style={{ background: 'rgba(30,64,175,0.08)', borderColor: 'rgba(30,64,175,0.2)', color: '#1E40AF' }}>
                <span style={{ color: '#F59E0B' }}>✦</span>
                Analysis Complete · {result.filename} · {result.duration_sec}s
              </div>
            </motion.div>

            {/* ── SIDE BY SIDE: Transcript + Correction ── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 }}
              >
                <TranscriptCard whisper={result.whisper} darkMode={darkMode} />
              </motion.div>
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
              >
                <CorrectionCard
                  correction={result.correction}
                  originalText={result.whisper.text}
                  darkMode={darkMode}
                />
              </motion.div>
            </div>

            {/* ── VAD TIMELINE (full width below) ── */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="mb-5"
            >
              <VadTimeline
                vad_segments={result.vad_segments}
                duration_sec={result.duration_sec}
                darkMode={darkMode}
              />
            </motion.div>

            {/* ── STATS / TIMING (full width below) ── */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className={`rounded-2xl p-6 mb-6 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}
            >
              <p className={`text-xs font-semibold text-center mb-5 tracking-widest uppercase ${
                darkMode ? 'text-gray-500' : 'text-gray-400'
              }`}>
                Pipeline Stats
              </p>

              {/* Summary stats row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                {[
                  { label: 'Duration', value: `${result.duration_sec}s` },
                  { label: 'Words', value: result.whisper.word_count },
                  { label: 'Segments', value: result.whisper.segments?.length || 0 },
                  { label: 'VAD Segments', value: result.vad_segments?.length || 0 },
                ].map((s, i) => (
                  <div key={i} className={`rounded-xl p-4 text-center ${
                    darkMode ? 'bg-gray-800' : 'bg-white border border-gray-100'
                  }`}>
                    <p className="text-xl font-black" style={{ color: '#1E40AF', fontFamily: 'Syne, sans-serif' }}>
                      {s.value}
                    </p>
                    <p className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{s.label}</p>
                  </div>
                ))}
              </div>

              {/* Timing pills */}
              <TimingBar timing={result.timing} darkMode={darkMode} />
            </motion.div>

            {/* Reset */}
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={reset}
              className={`w-full py-3.5 rounded-2xl font-semibold text-sm border-2 transition-all ${
                darkMode
                  ? 'border-white/10 text-gray-400 hover:border-blue-700 hover:text-white'
                  : 'border-gray-200 text-gray-500 hover:border-blue-700 hover:text-blue-700'
              }`}
            >
              ↩ Analyze Another Presentation
            </motion.button>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}