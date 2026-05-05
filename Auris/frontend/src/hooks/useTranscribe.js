import { useState } from 'react'

export default function useTranscribe() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [currentStep, setCurrentStep] = useState(-1)

  const analyze = async () => {
    if (!file) return

    setStatus('loading')
    setError(null)
    setResult(null)
    setCurrentStep(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Step 0 → extracting audio (immediately)
      setCurrentStep(0)

      // Small delay to show step 0 visually before request fires
      await new Promise(r => setTimeout(r, 400))
      setCurrentStep(1) // Voice Analysis

      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        body: formData,
      })

      setCurrentStep(2) // Generating Feedback

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.message || `Server error: ${response.status}`)
      }

      const data = await response.json()

      // Small delay to show step 2 completing
      await new Promise(r => setTimeout(r, 600))

      setResult(data)
      setStatus('success')
      setCurrentStep(-1)
    } catch (err) {
      setError(err.message)
      setStatus('error')
      setCurrentStep(-1)
    }
  }

  const reset = () => {
    setFile(null)
    setStatus('idle')
    setResult(null)
    setError(null)
    setCurrentStep(-1)
  }

  return { file, setFile, status, result, error, currentStep, analyze, reset }
}