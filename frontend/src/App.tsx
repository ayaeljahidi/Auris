import { useState, useCallback } from 'react';
import { useLenis } from '@/hooks/useLenis';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import Hero from '@/sections/Hero';
import TranscriptTicker from '@/sections/TranscriptTicker';
import Features from '@/sections/Features';
import WaveformBanner from '@/sections/WaveformBanner';
import ProcessSteps from '@/sections/ProcessSteps';
import LiveRecording from '@/sections/LiveRecording';
import FAQ from '@/sections/FAQ';
import UploadPage from '@/sections/UploadPage';
import LivePage from '@/sections/LivePage';
import ResultsPage from '@/sections/ResultsPage';
import type { PageView, AnalysisResult } from '@/types';

export default function App() {
  const [currentView, setCurrentView] = useState<PageView>('home');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  useLenis();

  const handleNavigate = useCallback((view: PageView) => {
    setCurrentView(view);
    window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
  }, []);

  const handleResult = useCallback((result: AnalysisResult) => {
    setAnalysisResult(result);
  }, []);

  return (
    <div className="min-h-[100dvh] bg-auris-bg">
      <Navbar currentView={currentView} onNavigate={handleNavigate} />

      {currentView === 'home' && (
        <main>
          <Hero onNavigate={handleNavigate} />
          <div id="about">
            <TranscriptTicker />
          </div>
          <Features />
          <WaveformBanner />
          <ProcessSteps />
          <LiveRecording onNavigate={handleNavigate} />
          <FAQ />
        </main>
      )}

      {currentView === 'upload' && (
        <UploadPage onNavigate={handleNavigate} onResult={handleResult} />
      )}

      {currentView === 'live' && (
        <LivePage onNavigate={handleNavigate} onResult={handleResult} />
      )}

      {currentView === 'results' && analysisResult && (
        <ResultsPage result={analysisResult} onNavigate={handleNavigate} />
      )}

      {currentView === 'home' && <Footer onNavigate={handleNavigate} />}
    </div>
  );
}
