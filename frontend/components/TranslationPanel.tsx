import React from 'react';
import { classifyWord } from '@/utils/api';

type WordResult = {
    original: string;
    translated: string;
};

type Props = {
    loading: boolean;
    error: string | null;
    fullTranslation: string | null;
    words: WordResult[];
    context: string | null;
    lang: string;
};

export default function TranslationPanel({ loading, error, fullTranslation, words, context, lang }: Props) {

    const handleWordClick = async (word: string, currentTarget: EventTarget & HTMLDivElement) => {
        const cleanWord = word.replace(/[:;.,!?"\)\(/=£$'*´`§<>_\-]/g, "");
        if (!cleanWord) return;

        // Visual feedback
        // Note: In React we usually drive style by state, but for a quick migration of this specific logic:
        const card = currentTarget;
        card.style.borderColor = '#f1c40f';
        card.style.backgroundColor = '#fff9c4';

        try {
            const data = await classifyWord(cleanWord, context || "", lang);
            console.log("Classification result:", data);

            card.style.borderColor = '#2ed573';
            card.style.backgroundColor = '#e8f5e9';
            card.title = `Saved as ${data.classification.part_of_speech}`;

            // We could update the translation here if the API returns a better one
            // but that would require local state management for the word list
        } catch (err) {
            console.error(err);
            card.style.borderColor = '#e74c3c';
        }
    };

    return (
        <div className="panel-section" id="translationPanel">
            {loading && <div style={{ textAlign: 'center', color: '#999' }}>Translating...</div>}

            {!loading && !fullTranslation && !error && (
                <div style={{ color: '#999', fontStyle: 'italic', textAlign: 'center', marginTop: '20px' }}>
                    Translations will appear here...
                </div>
            )}

            {error && <div style={{ color: 'red' }}>Error: {error}</div>}

            {fullTranslation && (
                <>
                    <div className="full-translation-box">
                        {fullTranslation}
                    </div>

                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {words.map((item, idx) => {
                            if (!item.original.trim()) return null;
                            return (
                                <div
                                    key={idx}
                                    className="word-card"
                                    onClick={(e) => handleWordClick(item.original, e.currentTarget)}
                                >
                                    <span className="word-original">{item.original}</span>
                                    <span className="word-translation">{item.translated}</span>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
}
