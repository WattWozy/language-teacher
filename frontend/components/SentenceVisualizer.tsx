import React, { useState } from 'react';

type Token = {
    text: string;
    lemma: string;
    pos: string;
    dep: string;
    head?: number;
    tense?: string;
    person?: string;
};

type Sentence = {
    text: string;
    tokens: Token[];
};

type Props = {
    sentence: Sentence;
};

export default function SentenceVisualizer({ sentence }: Props) {
    const [activeToken, setActiveToken] = useState<Token | null>(null);
    const [toggles, setToggles] = useState({
        tokens: true,
        tree: true,
        timeline: true,
        slots: true,
        morph: true
    });

    const toggle = (key: keyof typeof toggles) => {
        setToggles(prev => ({ ...prev, [key]: !prev[key] }));
    };

    // Render Tree Logic
    const renderTree = () => {
        const tokens = sentence.tokens;
        const width = 1000; // viewbox width
        const height = 300;
        const step = width / (tokens.length + 1);
        const textY = height - 40;

        const arcs = tokens.map((t, i) => {
            if (t.head === null || t.head === undefined) return null;

            const startX = (t.head + 1) * step;
            const endX = (i + 1) * step;
            const dist = Math.abs(t.head - i);
            const arcHeight = 40 + (dist * 20);
            const midX = (startX + endX) / 2;
            const midY = textY - arcHeight;

            return (
                <g key={`arc-${i}`}>
                    <path
                        d={`M ${startX} ${textY - 20} Q ${midX} ${midY} ${endX} ${textY - 20}`}
                        fill="none"
                        stroke="#5352ed"
                        strokeWidth="1.5"
                        markerEnd="url(#arrowhead)"
                        style={{ opacity: 0.8 }}
                    />
                    <text
                        x={midX}
                        y={midY - 5}
                        textAnchor="middle"
                        className="tree-label"
                    >
                        {t.dep}
                    </text>
                </g>
            );
        });

        const words = tokens.map((t, i) => {
            const x = (i + 1) * step;
            return (
                <g key={`word-${i}`}>
                    <text x={x} y={textY} textAnchor="middle" className="tree-text" fontWeight="bold">{t.text}</text>
                    <text x={x} y={textY + 20} textAnchor="middle" fill="#95a5a6" fontSize="12px">{t.pos}</text>
                </g>
            );
        });

        return (
            <svg viewBox={`0 0 ${width} ${height}`}>
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#5352ed" />
                    </marker>
                </defs>
                {arcs}
                {words}
            </svg>
        );
    };

    return (
        <div>
            <div className="section">
                <div className="section-title">View Toggles</div>
                <div className="toggle-group">
                    <label><input type="checkbox" checked={toggles.tokens} onChange={() => toggle('tokens')} /> Tokens</label>
                    <label><input type="checkbox" checked={toggles.tree} onChange={() => toggle('tree')} /> Dependency Tree</label>
                    <label><input type="checkbox" checked={toggles.timeline} onChange={() => toggle('timeline')} /> Tense Timeline</label>
                    <label><input type="checkbox" checked={toggles.slots} onChange={() => toggle('slots')} /> Slots</label>
                    <label><input type="checkbox" checked={toggles.morph} onChange={() => toggle('morph')} /> Morphology</label>
                </div>
            </div>

            {toggles.tokens && (
                <div id="tokens" className="section">
                    <div className="section-title">Sentence Tokens</div>
                    <div className="tokens">
                        {sentence.tokens.map((t, i) => (
                            <div
                                key={i}
                                className={`token role-${t.dep === 'root' ? 'VERB' : t.dep} ${activeToken === t ? 'active' : ''}`}
                                onClick={() => setActiveToken(t)}
                            >
                                {t.text}
                            </div>
                        ))}
                    </div>
                    {activeToken && (
                        <div className="panel">
                            <b>{activeToken.text}</b><br />
                            Lemma: {activeToken.lemma}<br />
                            POS: {activeToken.pos}<br />
                            Tense: {activeToken.tense || '-'}<br />
                            Person: {activeToken.person || '-'}
                        </div>
                    )}
                </div>
            )}

            {toggles.tree && (
                <div id="tree" className="section">
                    <div className="section-title">Dependency Tree</div>
                    {renderTree()}
                </div>
            )}

            {toggles.timeline && (
                <div id="timeline" className="section">
                    <div className="section-title">Tense / Aspect</div>
                    <div className="timeline">
                        <div
                            className="timeline-bar"
                            style={{
                                width: '30%',
                                left: sentence.tokens.find(t => t.dep === 'root')?.tense === 'PAST' ? '0%' :
                                    sentence.tokens.find(t => t.dep === 'root')?.tense === 'FUTURE' ? '60%' : '30%'
                            }}
                        ></div>
                    </div>
                </div>
            )}

            {toggles.slots && (
                <div id="slots" className="section">
                    <div className="section-title">Sentence Slots</div>
                    <div className="panel">
                        {(() => {
                            const subj = sentence.tokens.find(t => t.dep === 'nsubj');
                            const verb = sentence.tokens.find(t => t.dep === 'root');
                            const obj = sentence.tokens.find(t => t.dep === 'obj');
                            return (
                                <>
                                    [SUBJECT] {subj?.text || '—'}<br />
                                    [VERB] {verb?.text || '—'}<br />
                                    [OBJECT] {obj?.text || '—'}
                                </>
                            );
                        })()}
                    </div>
                </div>
            )}

            {toggles.morph && (
                <div id="morph" className="section">
                    <div className="section-title">Morphology</div>
                    <div className="panel">
                        {activeToken ? (
                            <>
                                <b>{activeToken.text}</b><br />
                                {activeToken.lemma} + <i>(morphology TODO)</i>
                            </>
                        ) : 'Click a token'}
                    </div>
                </div>
            )}
        </div>
    );
}
