import { useState } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Upload from './pages/Upload'

export default function App() {
  const [darkMode, setDarkMode] = useState(true)
  const location = useLocation()

  const toggleDarkMode = () => setDarkMode(prev => !prev)

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className={`min-h-screen transition-colors duration-500 ${
        darkMode ? 'bg-gray-950 text-white' : 'bg-white text-gray-900'
      }`}>
        <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Landing darkMode={darkMode} />} />
            <Route path="/upload" element={<Upload darkMode={darkMode} />} />
          </Routes>
        </AnimatePresence>
      </div>
    </div>
  )
}