import React from 'react';

type Props = {
    state: 'ready' | 'recording' | 'processing' | 'playing';
};

export default function VoiceVisualizer({ state }: Props) {
    let icon = "🎤";
    let statusText = "Ready";
    let statusClass = "";

    switch (state) {
        case 'recording':
            icon = "🎙️";
            statusText = "Listening...";
            statusClass = "state-recording";
            break;
        case 'processing':
            icon = ""; // Uses HTML for dots
            statusText = "Thinking...";
            statusClass = "state-processing";
            break;
        case 'playing':
            icon = "🔊";
            statusText = "Speaking...";
            statusClass = "state-playing";
            break;
        default:
            icon = "🎤";
            statusText = "Ready";
    }

    return (
        <div className={statusClass}>
            <div className="visualizer" id="icon">
                {state === 'processing' ? (
                    <div className="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                ) : icon}
            </div>
            <div className="status-text" id="status">{statusText}</div>
            <div className="instruction">Hold Spacebar to Talk</div>
        </div>
    );
}
