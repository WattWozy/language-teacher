'use client';

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useWebSocket } from '@/hooks/useWebSocket';
import { translateSentence } from '@/utils/api';
import VoiceVisualizer from '@/components/VoiceVisualizer';
import ChatInput from '@/components/ChatInput';
import ConversationLog from '@/components/ConversationLog';
import TranslationPanel from '@/components/TranslationPanel';

export default function Home() {
  const { status, messages, sendMessage, sendAudio, audioQueue, setAudioQueue } = useWebSocket('ws://localhost:8000/ws');
  const [uiState, setUiState] = useState<'ready' | 'recording' | 'processing' | 'playing'>('ready');

  // Translation State
  const [transLoading, setTransLoading] = useState(false);
  const [transError, setTransError] = useState<string | null>(null);
  const [fullTrans, setFullTrans] = useState<string | null>(null);
  const [words, setWords] = useState<any[]>([]);
  const [context, setContext] = useState<string | null>(null);
  const [targetLang, setTargetLang] = useState('pl'); // Default, maybe detect from last message

  // Audio Refs
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const isPlaying = useRef(false);

  // Effect to play audio queue
  useEffect(() => {
    const playNext = () => {
      if (audioQueue.length === 0) {
        isPlaying.current = false;
        setUiState('ready'); // Back to ready when queue empty
        return;
      }

      isPlaying.current = true;
      setUiState('playing');

      const nextAudio = audioQueue[0];
      const audio = new Audio(nextAudio);

      audio.onended = () => {
        isPlaying.current = false;
        setAudioQueue(prev => {
          const next = prev.slice(1);
          if (next.length === 0) setUiState('ready');
          return next;
        });
      };

      audio.play().catch(e => {
        console.error("Playback failed", e);
        isPlaying.current = false;
        setAudioQueue(prev => {
          const next = prev.slice(1);
          if (next.length === 0) setUiState('ready');
          return next;
        });
      });
    };

    if (!isPlaying.current && audioQueue.length > 0) {
      playNext();
    }
  }, [audioQueue, setAudioQueue]);

  // Keyboard (Spacebar) Logic
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if ((e.target as HTMLElement).tagName === 'INPUT') return;

      if (e.code === 'Space' && uiState === 'ready' && !e.repeat) {
        e.preventDefault(); // Prevent scrolling
        startRecording();
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;

      if (e.code === 'Space' && uiState === 'recording') {
        stopRecording();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [uiState]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (e) => {
        audioChunks.current.push(e.data);
      };

      mediaRecorder.current.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/wav' });
        sendAudio(blob);
        setUiState('processing');
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.current.start();
      setUiState('recording');
    } catch (err) {
      console.error("Mic Error:", err);
      alert("Could not access microphone");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
      mediaRecorder.current.stop();
    }
  };

  const handleSendMessage = (text: string) => {
    sendMessage(text);
    setUiState('processing');
  };

  const handleAnalyze = async (text: string, lang: string) => {
    setTransLoading(true);
    setTransError(null);
    setFullTrans(null);
    setContext(text);
    setTargetLang(lang);

    try {
      const data = await translateSentence(text, lang);
      setFullTrans(data.full_translation);
      setWords(data.words);
    } catch (e: any) {
      setTransError(e.message);
    } finally {
      setTransLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', minHeight: 'calc(100vh - 80px)', width: 'auto', gap: '40px' }}>
      <div className={`container ${uiState === 'recording' ? 'state-recording' : ''}`} id="app">
        <VoiceVisualizer state={uiState} />
        <ChatInput onSend={handleSendMessage} />
      </div>

      <div className="side-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div className="panel-title" style={{ marginBottom: 0 }}>Conversation</div>
          <Link href={`/words?lang=${targetLang}`} className="text-xs bg-green-500 text-white px-2 py-1 rounded hover:bg-green-600 no-underline" style={{ background: '#2ed573', textDecoration: 'none', fontSize: '0.8rem', color: 'white', padding: '4px 8px', borderRadius: '4px' }}>
            View Words
          </Link>
        </div>

        <ConversationLog messages={messages} onMessageClick={handleAnalyze} />

        <div className="panel-title">Translations</div>
        <TranslationPanel
          loading={transLoading}
          error={transError}
          fullTranslation={fullTrans}
          words={words}
          context={context}
          lang={targetLang}
        />
      </div>

      <div style={{ position: 'fixed', top: 10, left: 10, fontSize: '0.8rem', color: '#ccc' }}>
        Status: {status}
      </div>
    </div>
  );
}
