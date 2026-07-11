import sqlite3
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from .reminder_scheduler import schedule_reminder
DB_NAME = "healthcare.db"

scheduler = BackgroundScheduler()
scheduler.start()


def save_reminder(medicine, reminder_time, days):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reminders (medicine, reminder_time, days)
        VALUES (?, ?, ?)
    """, (medicine, reminder_time, days))

    conn.commit()
    conn.close()


def send_reminder(medicine, reminder_time):
    print(f"🔔 Reminder: Take {medicine} at {reminder_time}")


def schedule_reminder(medicine, reminder_time, days):

    hour = int(reminder_time.split(":")[0])
    minute = int(reminder_time.split(":")[1])

    end_date = datetime.now() + timedelta(days=days)

    scheduler.add_job(
        send_reminder,
        'cron',
        hour=hour,
        minute=minute,
        end_date=end_date,
        args=[medicine, reminder_time]
    )


def generate_reminders(medicine, times_per_day, days):
    reminders = []

    interval = 24 // times_per_day

    for i in range(times_per_day):
        hour = (8 + i * interval) % 24
        reminder_time = f"{hour:02d}:00"

        # Save to database
        save_reminder(medicine, reminder_time, days)

        # Schedule reminder
        schedule_reminder(medicine, reminder_time, days)

        reminders.append(reminder_time)

    return {
        "medicine": medicine,
        "reminder_times": reminders,
        "days": days
    }
