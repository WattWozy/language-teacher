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

    const activeSentence = sentences[selectedSentenceIndex];

    return (
        <div className="container" style={{ maxWidth: '1200px', width: '95%' }}>
            <Link href="/" className="back-btn">← Back to Assistant</Link>
            <h1>Grammar Inspector</h1>

            <div className="controls">
                <label htmlFor="langSelect" style={{ fontWeight: 'bold', marginRight: '5px' }}>Language:</label>
                <select
                    id="langSelect"
                    style={{ marginRight: '20px' }}
                    value={selectedLang}
                    onChange={(e) => setSelectedLang(e.target.value)}
                >
                    {languages.map(l => (
                        <option key={l} value={l}>{l.toUpperCase()}</option>
                    ))}
                </select>

                <label htmlFor="sentenceSelect" style={{ fontWeight: 'bold', marginRight: '5px' }}>Sentence:</label>
                <select
                    id="sentenceSelect"
                    value={selectedSentenceIndex}
                    onChange={(e) => setSelectedSentenceIndex(Number(e.target.value))}
                >
                    {sentences.map((s, i) => (
                        <option key={i} value={i}>{s.text}</option>
                    ))}
                </select>
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
