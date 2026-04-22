import { motion } from 'framer-motion'

const steps = [
  { label: 'Extracting Audio', icon: '🎬' },
  { label: 'Voice Analysis', icon: '🎙️' },
  { label: 'Generating Feedback', icon: '✨' },
]

export default function PipelineSteps({ currentStep, darkMode }) {
  return (
    <div className="py-12 flex flex-col items-center gap-10">
      {/* Waveform loader */}
      <div className="flex items-end gap-1 h-12">
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={i}
            className="w-1.5 rounded-full"
            style={{ background: '#1E40AF' }}
            animate={{ scaleY: [0.3, 1, 0.3] }}
            transition={{
              duration: 0.9,
              repeat: Infinity,
              delay: i * 0.08,
              ease: 'easeInOut',
            }}
          />
        ))}
      </div>

      {/* Steps */}
      <div className="flex items-center gap-0">
        {steps.map((step, i) => {
          const isComplete = currentStep > i
          const isActive = currentStep === i

          return (
            <div key={i} className="flex items-center">
              <div className="flex flex-col items-center gap-2">
                {/* Circle */}
                <div className="relative">
                  {isActive && (
                    <motion.div
                      className="absolute inset-0 rounded-full pulse-ring"
                      style={{ background: 'rgba(30,64,175,0.3)' }}
                      animate={{ scale: [1, 1.8], opacity: [0.6, 0] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                    />
                  )}
                  <motion.div
                    animate={{
                      background: isComplete ? '#F59E0B' : isActive ? '#1E40AF' : 'transparent',
                      borderColor: isComplete ? '#F59E0B' : isActive ? '#1E40AF' : '#6B7280',
                    }}
                    className="w-10 h-10 rounded-full border-2 flex items-center justify-center text-lg relative z-10"
                  >
                    {isComplete ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                      </svg>
                    ) : (
                      <span className={`text-base ${isActive ? 'text-white' : 'text-gray-500'}`}>
                        {step.icon}
                      </span>
                    )}
                  </motion.div>
                </div>

                {/* Label */}
                <span className={`text-xs font-semibold w-24 text-center leading-tight ${
                  isComplete ? 'text-yellow-400' : isActive
                    ? darkMode ? 'text-white' : 'text-gray-900'
                    : 'text-gray-500'
                }`}
                style={{ color: isComplete ? '#F59E0B' : isActive ? (darkMode ? '#fff' : '#111') : undefined }}>
                  {step.label}
                </span>
              </div>

              {/* Connector line */}
              {i < steps.length - 1 && (
                <div className="w-20 h-0.5 mx-2 mb-6 rounded-full overflow-hidden bg-gray-700 relative">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: '#F59E0B' }}
                    animate={{ width: currentStep > i ? '100%' : '0%' }}
                    transition={{ duration: 0.5, ease: 'easeInOut' }}
                  />
                </div>
              )}
            </div>
          )
        })}
      </div>

      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        {currentStep === 0 && 'Converting your file with FFmpeg…'}
        {currentStep === 1 && 'Silero VAD + Whisper analyzing your speech…'}
        {currentStep === 2 && 'Flan-T5 generating corrections and feedback…'}
      </p>
    </div>
  )
}