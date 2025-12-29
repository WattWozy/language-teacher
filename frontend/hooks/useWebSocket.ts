import { useEffect, useRef, useState, useCallback } from 'react';

type Message = {
    type: 'log' | 'audio';
    role?: 'user' | 'assistant' | 'system';
    text?: string;
    lang?: string;
    data?: string;
};

export function useWebSocket(url: string) {
    const ws = useRef<WebSocket | null>(null);
    const [status, setStatus] = useState<string>('Connecting...');
    const [messages, setMessages] = useState<Message[]>([]);
    const [audioQueue, setAudioQueue] = useState<string[]>([]);

    useEffect(() => {
        ws.current = new WebSocket(url);

        ws.current.onopen = () => {
            setStatus('Ready');
        };

        ws.current.onclose = () => {
            setStatus('Disconnected');
        };

        ws.current.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'audio') {
                    const audioBase64 = message.data;
                    const audioBytes = Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0));
                    const audioBlob = new Blob([audioBytes], { type: "audio/wav" });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    setAudioQueue(prev => [...prev, audioUrl]);
                } else if (message.type === 'log') {
                    setMessages(prev => [...prev, message]);
                }
            } catch (e) {
                console.error("Error parsing message:", e);
            }
        };

        return () => {
            ws.current?.close();
        };
    }, [url]);

    const sendMessage = useCallback((text: string) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(text);
        }
    }, []);

    const sendAudio = useCallback((audioBlob: Blob) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            audioBlob.arrayBuffer().then(buffer => {
                ws.current?.send(buffer);
            });
        }
    }, []);

    return { status, messages, sendMessage, sendAudio, audioQueue, setAudioQueue };
}
