import sqlite3

def get_connection():
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        confidence REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    return conn

def save_prediction(filename, prediction, confidence):
    conn = get_connection()
    conn.execute("INSERT INTO predictions (filename, prediction, confidence) VALUES (?, ?, ?)",
                 (filename, prediction, confidence))
    conn.commit()
    conn.close()

def get_history():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return rows
