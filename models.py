from sqlalchemy import Column, Integer, String, DateTime
from database1 import Base
from datetime import datetime


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(String, default="guest")

    user_message = Column(String)

    ai_response = Column(String)

    timestamp = Column(DateTime, default=datetime.utcnow)