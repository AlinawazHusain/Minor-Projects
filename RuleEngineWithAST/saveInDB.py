import sqlite3

def init_db():
    conn = sqlite3.connect('rules.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def save_rule(rule: str):
    conn = sqlite3.connect('rules.db')
    c = conn.cursor()
    c.execute("INSERT INTO rules (rule) VALUES (?)", (rule,))
    conn.commit()
    conn.close()