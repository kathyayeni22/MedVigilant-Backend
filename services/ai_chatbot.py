from groq import Groq
from database1 import SessionLocal
from models import ChatHistory
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
# -------------------------
# GROQ CLIENT
# -------------------------
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# -------------------------
# GET CHAT MEMORY
# -------------------------
def get_chat_memory(user_id, limit=5):

    db = SessionLocal()

    chats = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
        .all()
    )

    db.close()

    memory = []

    for chat in reversed(chats):
        memory.append({
            "role": "user",
            "content": chat.user_message
        })

        memory.append({
            "role": "assistant",
            "content": chat.ai_response
        })

    return memory


# -------------------------
# AI RESPONSE
# -------------------------
def get_ai_response(message, user_id="guest"):

    try:

        memory = get_chat_memory(user_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a smart AI healthcare assistant. "
                    "Give short, caring, useful health advice. "
                    "If user feels stressed, anxious or sad, "
                    "give emotional support."
                )
            }
        ]

        messages.extend(memory)

        messages.append({
            "role": "user",
            "content": message
        })
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        ai_response = completion.choices[0].message.content

        # -------------------------
        # SAVE CHAT
        # -------------------------
        db = SessionLocal()

        chat = ChatHistory(
            user_id=user_id,
            user_message=message,
            ai_response=ai_response,
            timestamp=datetime.utcnow()
        )

        db.add(chat)
        db.commit()
        db.close()

        return ai_response

    except Exception as e:
        print("AI ERROR:", e)
        return "Sorry, AI service is temporarily unavailable."


# -------------------------
# GET CHAT HISTORY
# -------------------------
def get_chat_history(user_id="guest"):

    db = SessionLocal()

    chats = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.timestamp.asc())
        .all()
    )

    db.close()

    result = []

    for c in chats:
        result.append({
            "user_message": c.user_message,
            "ai_response": c.ai_response,
            "timestamp": str(c.timestamp)
        })

    return result


# -------------------------
# CLEAR CHAT
# -------------------------
def clear_chat_history(user_id):

    db = SessionLocal()

    db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id
    ).delete()

    db.commit()
    db.close()

    return {"message": "Chat history cleared"}