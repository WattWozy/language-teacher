const API_BASE = 'http://localhost:8000';

export async function translateSentence(text: string, fromLang: string = 'pl') {
    const response = await fetch(`${API_BASE}/translate/sentence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text,
            from_lang: fromLang,
            to_lang: 'en'
        })
    });
    if (!response.ok) throw new Error('Translation failed');
    return response.json();
}

export async function classifyWord(word: string, context: string, lang: string) {
    const response = await fetch(`${API_BASE}/roots/${lang}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            word,
            context
        })
    });
    if (!response.ok) throw new Error('Classification failed');
    return response.json();
}
