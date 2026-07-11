from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

scheduler = BackgroundScheduler()
scheduler.start()


def send_reminder(medicine: str, reminder_time: str):
    """
    This function is called whenever a reminder is triggered.
    For now, it prints a notification.
    Later, this can be replaced with WebSocket to notify the app.
    """
    print(f"🔔 Reminder: Take {medicine} at {reminder_time}")


def schedule_reminder(medicine: str, reminder_time: str, days: int):
    """
    Schedule a reminder for the given medicine, time, and number of days.
    """

    hour, minute = map(int, reminder_time.split(":"))

    # End date for the scheduler job
    end_date = datetime.now() + timedelta(days=days)

    scheduler.add_job(
        send_reminder,
        'cron',
        hour=hour,
        minute=minute,
        end_date=end_date,
        args=[medicine, reminder_time]
    )
