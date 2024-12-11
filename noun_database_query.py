# query.py
import sqlite3
from pypinyin import pinyin, Style
from difflib import SequenceMatcher

def word_to_pinyin(word):
    phones = pinyin(word, style=Style.TONE3)
    return ' '.join([p[0] for p in phones])

def get_similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def query_word(query_word, threshold=0.8):
    conn = sqlite3.connect('words.db')
    c = conn.cursor()
    
    query_phoneme = word_to_pinyin(query_word)
    
    # Get all words and phonemes
    c.execute('SELECT word, phoneme FROM words')
    results = c.fetchall()
    
    best_match = None
    best_score = 0
    
    for word, phoneme in results:
        score = get_similarity(query_phoneme, phoneme)
        if score > best_score:
            best_score = score
            best_match = (word, phoneme)
    
    conn.close()
    
    if best_match:
        return {
            'query_word': query_word,
            'query_phoneme': query_phoneme,
            'matched_word': best_match[0],
            'matched_phoneme': best_match[1],
            'score': best_score,
            'is_active': best_score >= threshold
        }
    return None

# Example usage
if __name__ == "__main__":
    result = query_word("你好")
    print(f"Query result: {result}")