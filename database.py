import sqlite3

DB_NAME = "healthcare.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine TEXT,
            reminder_time TEXT,
            days INTEGER
        )
    """)

    conn.commit()
    conn.close()


def save_reminder(medicine, reminder_time, days):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reminders (medicine, reminder_time, days)
        VALUES (?, ?, ?)
    """, (medicine, reminder_time, days))

    conn.commit()
    conn.close()
