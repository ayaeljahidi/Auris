import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export default function Navbar({ darkMode, toggleDarkMode }) {
  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 glass border-b ${
        darkMode
          ? 'bg-gray-950/80 border-white/5'
          : 'bg-white/80 border-gray-200'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg bg-blue-700 flex items-center justify-center">
            <span className="text-gold-400 font-bold text-sm" style={{ color: '#F59E0B' }}>A</span>
          </div>
          <span className="font-display text-xl font-800 tracking-tight">
            <span className="text-blue-700" style={{ color: '#1E40AF', fontFamily: 'Syne, sans-serif', fontWeight: 800 }}>Au</span>
            <span style={{ color: '#F59E0B', fontFamily: 'Syne, sans-serif', fontWeight: 800 }}>ris</span>
          </span>
        </Link>

        {/* Dark mode toggle */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={toggleDarkMode}
          className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
            darkMode
              ? 'bg-white/10 hover:bg-white/20 text-yellow-400'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
          }`}
        >
          {darkMode ? (
            // Sun icon
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5"/>
              <line x1="12" y1="1" x2="12" y2="3"/>
              <line x1="12" y1="21" x2="12" y2="23"/>
              <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
              <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
              <line x1="1" y1="12" x2="3" y2="12"/>
              <line x1="21" y1="12" x2="23" y2="12"/>
              <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
              <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
            </svg>
          ) : (
            // Moon icon
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
          )}
        </motion.button>
      </div>
    </motion.nav>
  )
}