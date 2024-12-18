import json
from pypinyin import pinyin, Style
from difflib import SequenceMatcher
import sqlite3

def get_text(data):
    ret = []
    for i, word_entry in enumerate(data):
        ret.append((word_entry["word"], i))
    return ret

def word_to_pinyin(word):
    phones = pinyin(word, style=Style.TONE3)
    return ''.join([p[0] for p in phones])

def create_db():
    conn = sqlite3.connect('words.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS words
                 (word TEXT PRIMARY KEY,
                  phoneme TEXT,
                  word_index INTEGER)''')
    return conn, c

def populate_db(data_file):
    conn, c = create_db()
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    words_with_indices = get_text(data)

    for word, index in words_with_indices:
        phoneme = word_to_pinyin(word)
        c.execute('INSERT OR REPLACE INTO words VALUES (?, ?, ?)', 
                 (word, phoneme, index))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    populate_db("../raw_dataset/text.json")