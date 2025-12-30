'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import SentenceVisualizer from '@/components/SentenceVisualizer';

export default function SentencesPage() {
    const [languages, setLanguages] = useState<string[]>([]);
    const [selectedLang, setSelectedLang] = useState<string>('');
    const [sentences, setSentences] = useState<any[]>([]);
    const [selectedSentenceIndex, setSelectedSentenceIndex] = useState<number>(0);
    const [loading, setLoading] = useState(true);
    const [customText, setCustomText] = useState('');
    const [customSentence, setCustomSentence] = useState<any>(null);
    const [mode, setMode] = useState<'browse' | 'custom'>('browse');

    // Initial Load - Languages
    useEffect(() => {
        async function fetchLanguages() {
            try {
                const res = await fetch('http://localhost:8000/sentences');
                const langs = await res.json();
                setLanguages(langs);
                if (langs.length > 0) {
                    setSelectedLang(langs[0]);
                } else {
                    setLoading(false);
                }
            } catch (e) {
                console.error("Failed to fetch languages", e);
                setLoading(false);
            }
        }
        fetchLanguages();
    }, []);

    // Load Sentences when Lang changes
    useEffect(() => {
        if (!selectedLang) return;

        async function fetchSentences() {
            setLoading(true);
            try {
                const res = await fetch(`http://localhost:8000/sentences/${selectedLang}.json`);
                const data = await res.json();
                setSentences(data);
                setSelectedSentenceIndex(0);
            } catch (e) {
                console.error("Failed to fetch sentences", e);
            } finally {
                setLoading(false);
            }
        }
        fetchSentences();
    }, [selectedLang]);

    const handleAnalyze = async () => {
        if (!customText.trim() || !selectedLang) return;
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/analyze/sentence', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: customText, lang: selectedLang })
            });
            const data = await res.json();
            setCustomSentence(data);
            setMode('custom');
        } catch (e) {
            console.error("Failed to analyze sentence", e);
        } finally {
            setLoading(false);
        }
    };

    const activeSentence = mode === 'browse' ? sentences[selectedSentenceIndex] : customSentence;

    return (
        <div className="container" style={{ maxWidth: '1200px', width: '95%' }}>
            <Link href="/" className="back-btn">← Back to Assistant</Link>
            <h1>Grammar Inspector</h1>

            <div className="controls" style={{ display: 'flex', gap: '20px', alignItems: 'flex-end', marginBottom: '30px' }}>
                <div>
                    <label htmlFor="langSelect" style={{ fontWeight: 'bold', display: 'block', marginBottom: '5px' }}>Language:</label>
                    <select
                        id="langSelect"
                        value={selectedLang}
                        onChange={(e) => setSelectedLang(e.target.value)}
                        style={{ padding: '8px', borderRadius: '5px' }}
                    >
                        {languages.map(l => (
                            <option key={l} value={l}>{l.toUpperCase()}</option>
                        ))}
                    </select>
                </div>

                <div style={{ flex: 1, borderLeft: '1px solid #ddd', paddingLeft: '20px' }}>
                    <label style={{ fontWeight: 'bold', display: 'block', marginBottom: '5px' }}>Browse Saved:</label>
                    <select
                        id="sentenceSelect"
                        value={selectedSentenceIndex}
                        onChange={(e) => {
                            setSelectedSentenceIndex(Number(e.target.value));
                            setMode('browse');
                        }}
                        style={{ width: '100%', padding: '8px', borderRadius: '5px' }}
                    >
                        {sentences.map((s, i) => (
                            <option key={i} value={i}>{s.text}</option>
                        ))}
                    </select>
                </div>

                <div style={{ flex: 2, borderLeft: '1px solid #ddd', paddingLeft: '20px' }}>
                    <label htmlFor="customInput" style={{ fontWeight: 'bold', display: 'block', marginBottom: '5px' }}>Analyze New Sentence:</label>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            id="customInput"
                            type="text"
                            placeholder="Type a sentence..."
                            value={customText}
                            onChange={(e) => setCustomText(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                            style={{ flex: 1, padding: '8px', borderRadius: '5px', border: '1px solid #ccc' }}
                        />
                        <button
                            onClick={handleAnalyze}
                            className="btn-primary"
                            style={{ whiteSpace: 'nowrap' }}
                        >
                            Analyze
                        </button>
                    </div>
                </div>
            </div>

            {loading && <div style={{ textAlign: 'center' }}>Loading...</div>}

            {!loading && activeSentence && (
                <SentenceVisualizer sentence={activeSentence} />
            )}

            {!loading && sentences.length === 0 && (
                <div style={{ textAlign: 'center' }}>No sentences found for this language.</div>
            )}
        </div>
    );
}
