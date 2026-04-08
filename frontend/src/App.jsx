import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, 
  Download, 
  ChevronRight, 
  BrainCircuit, 
  Cpu, 
  Zap, 
  AlertCircle,
  BarChart3,
  Lightbulb,
  Layers
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { jsPDF } from 'jspdf';
import 'jspdf-autotable';

const App = () => {
  const [problem, setProblem] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);

  const recognitionRef = useRef(null);

  useEffect(() => {
    axios.get('/api/suggestions')
      .then(res => setSuggestions(res.data))
      .catch(() => {});

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      recognition.onresult = (e) => {
        setProblem(e.results[0][0].transcript);
        setIsRecording(false);
      };
      recognition.onerror = () => setIsRecording(false);
      recognition.onend = () => setIsRecording(false);
      recognitionRef.current = recognition;
    }
  }, []);

  const toggleRecording = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      setError(null);
      recognitionRef.current?.start();
      setIsRecording(true);
    }
  };

  const handlePredict = async () => {
    if (!problem.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await axios.post('/api/predict', { description: problem });
      setResult(response.data);
    } catch (err) {
      setError('Analysis failed. Please verify that the backend service is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!result) return;
    const doc = new jsPDF();

    // Dark header bar
    doc.setFillColor(15, 23, 42);
    doc.rect(0, 0, 210, 42, 'F');
    doc.setFillColor(99, 102, 241);
    doc.rect(0, 42, 210, 2, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('Paradigm AI  -  Strategic Blueprint', 20, 27);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(148, 163, 184);
    doc.text(`Generated ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}`, 20, 36);

    let y = 58;
    doc.setTextColor(100, 116, 139);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('BUSINESS PROBLEM', 20, y);
    y += 8;
    doc.setTextColor(30, 41, 59);
    doc.setFontSize(11);
    doc.setFont('helvetica', 'normal');
    const lines = doc.splitTextToSize(problem, 170);
    doc.text(lines, 20, y);
    y += lines.length * 6 + 12;

    // Classification box
    doc.setDrawColor(226, 232, 240);
    doc.setFillColor(248, 250, 252);
    doc.roundedRect(20, y, 170, 28, 4, 4, 'FD');
    doc.setFontSize(9);
    doc.setTextColor(100, 116, 139);
    doc.setFont('helvetica', 'bold');
    doc.text('PARADIGM', 30, y + 11);
    doc.text('CONFIDENCE', 30, y + 21);
    doc.setFontSize(11);
    doc.setTextColor(99, 102, 241);
    doc.text(result.ml_type.toUpperCase(), 80, y + 11);
    doc.setTextColor(30, 41, 59);
    doc.text(`${(result.confidence * 100).toFixed(1)}%`, 80, y + 21);
    y += 38;

    // Justification
    doc.setTextColor(100, 116, 139);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('STRATEGIC REASONING', 20, y);
    y += 7;
    doc.setTextColor(51, 65, 85);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    const justLines = doc.splitTextToSize(result.justification, 170);
    doc.text(justLines, 20, y);
    y += justLines.length * 5 + 12;

    // Algorithms
    doc.setTextColor(100, 116, 139);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text('RECOMMENDED ALGORITHMS', 20, y);
    y += 7;
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(30, 41, 59);
    result.algorithms.forEach((a) => {
      doc.text(`  \u2022  ${a}`, 22, y);
      y += 6;
    });
    y += 6;

    // Roadmap table
    doc.autoTable({
      startY: y,
      head: [['Phase', 'Action Item']],
      body: result.roadmap.map((s, i) => [`Step ${i + 1}`, s]),
      theme: 'grid',
      headStyles: { fillColor: [79, 70, 229], fontSize: 9, fontStyle: 'bold' },
      bodyStyles: { fontSize: 9 },
      margin: { left: 20, right: 20 },
      styles: { cellPadding: 5 }
    });

    doc.save(`Paradigm_Blueprint_${result.ml_type}.pdf`);
  };

  const confidencePercent = result ? (result.confidence * 100).toFixed(0) : 0;

  return (
    <div>
      <div className="mesh-bg"><div className="noise" /></div>

      <div className="app-container">
        {/* ---- Header ---- */}
        <motion.header
          className="header"
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="header-icon">
            <BrainCircuit size={28} />
          </div>
          <div>
            <h1 className="header-title">Paradigm AI</h1>
            <p className="header-sub">ML Decision Support System</p>
          </div>
        </motion.header>

        {/* ---- Input ---- */}
        <motion.section
          className="glass input-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.5 }}
        >
          <div className="input-section-header">
            <div className="input-section-title">
              <Cpu size={16} />
              <span>Describe Your Business Problem</span>
            </div>
            {isRecording && (
              <div className="recording-badge">
                <div className="recording-dot" />
                Recording
              </div>
            )}
          </div>

          <div className="input-wrapper">
            <textarea
              className="input-area"
              placeholder="e.g., Predict customer churn based on historical usage metrics and demographics..."
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handlePredict(); } }}
            />
            <button
              className={`mic-btn ${isRecording ? 'recording' : ''}`}
              onClick={toggleRecording}
              title="Voice input"
            >
              <Mic size={20} />
            </button>
          </div>

          <div className="action-bar">
            <div className="suggestions">
              {suggestions.slice(0, 3).map((s, i) => (
                <button key={i} className="suggestion-chip" onClick={() => setProblem(s)}>
                  {s.length > 40 ? s.substring(0, 38) + '...' : s}
                </button>
              ))}
            </div>
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={loading || !problem.trim()}
            >
              {loading ? <Zap size={18} className="spin" /> : <ChevronRight size={18} />}
              {loading ? 'Analyzing...' : 'Classify Problem'}
            </button>
          </div>

          {error && (
            <div className="error-box">
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </motion.section>

        {/* ---- Results ---- */}
        <AnimatePresence mode="wait">
          {result && (
            <motion.div
              className="results-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
            >
              {/* Meta Strip */}
              <div className="meta-strip">
                <div className="meta-left">
                  <span className={`badge badge-${result.ml_type.toLowerCase()}`}>
                    {result.ml_type}
                  </span>
                  <div>
                    <span className="confidence-value">{confidencePercent}</span>
                    <span className="confidence-unit">%</span>
                  </div>
                  <span className="confidence-label">Confidence Score</span>
                </div>
                <div className="meta-right">
                  <div className="justification-title">
                    <Lightbulb size={16} />
                    Strategic Reasoning
                  </div>
                  <p className="justification-text">{result.justification}</p>
                </div>
              </div>

              {/* Blueprint Panel */}
              <div className="blueprint">
                <div className="blueprint-header">
                  <div>
                    <div className="blueprint-title">Implementation Roadmap</div>
                    <div className="blueprint-sub">Phase-based execution plan tailored to your problem</div>
                  </div>
                  <button className="btn-outline" onClick={downloadReport}>
                    <Download size={16} />
                    Export PDF
                  </button>
                </div>

                <div className="blueprint-body">
                  {/* Algorithms */}
                  <div className="algo-sidebar">
                    <div className="algo-label">
                      <Layers size={12} />
                      Recommended Algorithms
                    </div>
                    <div className="algo-list">
                      {result.algorithms.map((algo, i) => (
                        <motion.div
                          key={i}
                          className="algo-item"
                          initial={{ opacity: 0, x: -12 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 + i * 0.07 }}
                        >
                          <span className="algo-number">{i + 1}</span>
                          {algo}
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* Roadmap */}
                  <div className="roadmap-panel">
                    <div className="roadmap-label">Execution Sequence</div>
                    <div className="roadmap-timeline">
                      {result.roadmap.map((step, i) => (
                        <motion.div
                          key={i}
                          className="roadmap-step"
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.2 + i * 0.1 }}
                        >
                          <div className="step-indicator">
                            <div className="step-dot">{i + 1}</div>
                            {i < result.roadmap.length - 1 && <div className="step-line" />}
                          </div>
                          <div className="step-body">
                            <p>{step}</p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ---- Footer ---- */}
        <footer className="footer">
          <div className="footer-icons">
            <Zap size={18} />
            <Cpu size={18} />
            <BarChart3 size={18} />
          </div>
          <p className="footer-text">&copy; 2026 Paradigm AI Decision Support System</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
