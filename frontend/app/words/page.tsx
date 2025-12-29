'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useSearchParams, useRouter } from 'next/navigation';

export default function WordsPage() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const langParam = searchParams.get('lang') || 'pl';
    const [lang, setLang] = useState(langParam);
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setLang(langParam);
    }, [langParam]);

    useEffect(() => {
        async function loadWords() {
            setLoading(true);
            setError(null);
            try {
                const res = await fetch(`http://localhost:8000/roots/${lang}.json`);
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                const json = await res.json();
                setData(json);
            } catch (e) {
                console.error(e);
                setError(`Error loading words for '${lang}'. Make sure the file exists.`);
                setData(null);
            } finally {
                setLoading(false);
            }
        }
        loadWords();
    }, [lang]);

    const handleLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const newLang = e.target.value;
        setLang(newLang);
        router.push(`/words?lang=${newLang}`);
    };

    return (
        <div className="container" style={{ maxWidth: '1200px', width: '95%' }}>
            <Link href="/" className="back-btn" style={{
                display: 'inline-flex', alignItems: 'center', marginBottom: '20px',
                textDecoration: 'none', color: '#5352ed', fontWeight: 'bold',
                padding: '10px 20px', background: 'rgba(83, 82, 237, 0.1)',
                borderRadius: '20px'
            }}>
                ← Back to Assistant
            </Link>

            <h1>Collected Words</h1>

            <div className="controls" style={{ marginBottom: '30px', textAlign: 'center' }}>
                <label htmlFor="langSelect" style={{ marginRight: '10px' }}>Language: </label>
                <select id="langSelect" value={lang} onChange={handleLangChange} style={{
                    padding: '10px 15px', fontSize: '16px', borderRadius: '20px',
                    border: '2px solid #e0e0e0', outline: 'none'
                }}>
                    <option value="pl">Polish (pl)</option>
                    <option value="en">English (en)</option>
                    <option value="it">Italian (it)</option>
                    <option value="no">Norwegian (no)</option>
                    <option value="uk">Ukrainian (uk)</option>
                </select>
            </div>

            <div id="content">
                {loading && <div style={{ textAlign: 'center' }}>Loading...</div>}
                {error && <div style={{ color: 'red', textAlign: 'center' }}>{error}</div>}
                {!loading && !error && (!data || Object.keys(data).length === 0) && (
                    <div style={{ textAlign: 'center' }}>No words collected yet.</div>
                )}

                {!loading && data && Object.entries(data).map(([pos, categories]: [string, any]) => (
                    <div key={pos} className="word-group" style={{ marginBottom: '40px' }}>
                        <div className="pos-title" style={{
                            fontSize: '1.5em', color: '#5352ed', borderBottom: '2px solid #eee',
                            paddingBottom: '10px', marginBottom: '20px', textTransform: 'uppercase',
                            fontWeight: 700
                        }}>
                            {pos}
                        </div>
                        {Object.entries(categories).map(([category, words]: [string, any]) => (
                            <div key={category}>
                                <div className="category-title" style={{
                                    fontSize: '1.1em', color: '#2ed573', margin: '20px 0 15px',
                                    fontWeight: 600, textTransform: 'uppercase'
                                }}>
                                    {category}
                                </div>
                                <div className="words-grid" style={{
                                    display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                                    gap: '20px'
                                }}>
                                    {Object.entries(words).map(([wordText, details]: [string, any]) => (
                                        <div key={wordText} className="word-card">
                                            <div className="word-main" style={{ fontSize: '1.3em', fontWeight: 'bold', color: '#2c3e50', marginBottom: '8px' }}>
                                                {wordText}
                                            </div>
                                            <div className="word-translation" style={{ color: '#5352ed', fontWeight: 500, marginBottom: '12px', fontSize: '1.1em' }}>
                                                {details.translation || '(No translation)'}
                                            </div>
                                            <div className="word-meta" style={{ fontSize: '0.85em', color: '#95a5a6' }}>
                                                {details.definition && <div>Def: {details.definition}</div>}
                                                {details.metadata?.lemma && <div>Lemma: {details.metadata.lemma}</div>}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                ))}
            </div>
        </div>
    );
}

// Suspense boundary requirement for component using useSearchParams in Next.js
export const dynamic = 'force-dynamic';
