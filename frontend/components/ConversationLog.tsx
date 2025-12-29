import React, { useEffect, useRef } from 'react';

type Message = {
    role?: 'user' | 'assistant' | 'system';
    text?: string;
    lang?: string;
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
        <div className="panel-section" id="conversationPanel">
            {messages.map((msg, idx) => (
                <div
                    key={idx}
                    className={`log-entry log-${msg.role}`}
                    title="Click to analyze words"
                    onClick={() => msg.text && onMessageClick(msg.text, msg.lang || 'en')}
                >
                    {msg.lang && (
                        <span style={{ fontSize: '0.7em', fontWeight: 'bold', color: '#888', marginRight: '5px' }}>
                            [{msg.lang.toUpperCase()}]
                        </span>
                    )}
                    <span>{msg.text}</span>
                </div>
            ))}
            <div ref={endRef} />
        </div>
    );
}
