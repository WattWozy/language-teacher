import React, { useEffect, useRef } from 'react';

type WordTranslation = {
    original: string;
    translated: string;
};

type Message = {
    role?: 'user' | 'assistant' | 'system';
    text?: string;
    lang?: string;
    translations?: {
        full: string;
        words: WordTranslation[];
    };
};

type Props = {
    messages: Message[];
    onMessageClick: (text: string, lang: string) => void;
};

export default function ConversationLog({ messages, onMessageClick }: Props) {
    const endRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="panel-section" id="conversationPanel" style={{ overflowX: 'hidden' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                {/* Header for clarity if needed, or just let the bubbles speak */}
                <div className="panel-title" style={{ fontSize: '0.7rem', color: '#888' }}>ORIGINAL</div>
                <div className="panel-title" style={{ fontSize: '0.7rem', color: '#888' }}>LITERAL / FULL</div>

                {messages.map((msg, idx) => {
                    const isAssistant = msg.role === 'assistant';

                    return (
                        <React.Fragment key={idx}>
                            {/* Left Column: Original */}
                            <div
                                className={`log-entry log-${msg.role}`}
                                style={{ margin: 0, height: 'fit-content' }}
                                title="Click to analyze words"
                                onClick={() => msg.text && onMessageClick(msg.text, msg.lang || 'en')}
                            >
                                {msg.lang && msg.role !== 'system' && (
                                    <span style={{ fontSize: '0.7em', fontWeight: 'bold', color: '#888', marginRight: '5px' }}>
                                        [{msg.lang.toUpperCase()}]
                                    </span>
                                )}
                                <span>{msg.text}</span>
                            </div>

                            {/* Right Column: Translation */}
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {isAssistant && msg.translations ? (
                                    <div className="log-entry" style={{
                                        margin: 0,
                                        backgroundColor: 'rgba(83, 82, 237, 0.05)',
                                        borderLeft: '4px solid #5352ed',
                                        fontSize: '0.9rem'
                                    }}>
                                        <div style={{ fontWeight: '600', color: '#5352ed' }}>
                                            {msg.translations.full}
                                        </div>
                                    </div>
                                ) : (
                                    msg.role === 'user' ? (
                                        <div style={{ color: '#ccc', fontStyle: 'italic', fontSize: '0.8rem', padding: '10px' }}>
                                            (User input)
                                        </div>
                                    ) : msg.role === 'system' ? (
                                        <div style={{ color: '#ccc', fontStyle: 'italic', fontSize: '0.8rem', padding: '10px' }}>
                                            (System)
                                        </div>
                                    ) : (
                                        <div style={{ color: '#ccc', fontStyle: 'italic', fontSize: '0.8rem', padding: '10px' }}>
                                            ...
                                        </div>
                                    )
                                )}
                            </div>
                        </React.Fragment>
                    );
                })}
            </div>
            <div ref={endRef} />
        </div>
    );
}
