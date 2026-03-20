import os
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DATABASE_URL, sslmode='require')


def get_session(session_id):
    try:
        conn = get_connection()
        cur = conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        cur.execute(
            "SELECT * FROM sessions WHERE session_id = %s",
            (session_id,)
        )
        row = cur.fetchone()

        if row:
            session = dict(row)
            intents_str = session.get("intents_seen", "")
            session["intents_seen"] = set(
                intents_str.split(",")
            ) if intents_str else set()
            cur.close()
            conn.close()
            return session
        else:
            cur.execute("""
                INSERT INTO sessions (session_id)
                VALUES (%s) RETURNING *
            """, (session_id,))
            conn.commit()
            row = cur.fetchone()
            session = dict(row)
            session["intents_seen"] = set()
            cur.close()
            conn.close()
            return session

    except Exception as e:
        print(f"DB get_session error: {e}")
        return {
            "session_id": session_id,
            "stage": "AWARENESS",
            "last_intent": None,
            "message_count": 0,
            "meaningful_message_count": 0,
            "cta_shown": False,
            "cta_shown_count": 0,
            "tone": "neutral",
            "email_provided": False,
            "phone_provided": False,
            "objection_count": 0,
            "intents_seen": set(),
            "lead_score": 1,
            "history": []
        }


def update_session(session_id, updates):
    try:
        conn = get_connection()
        cur = conn.cursor()

        if "intents_seen" in updates:
            if isinstance(updates["intents_seen"], set):
                updates["intents_seen"] = ",".join(
                    updates["intents_seen"]
                )

        updates["last_updated"] = datetime.now()

        set_clause = ", ".join([
            f"{k} = %s" for k in updates.keys()
        ])
        values = list(updates.values()) + [session_id]

        cur.execute(
            f"UPDATE sessions SET {set_clause} "
            f"WHERE session_id = %s",
            values
        )
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print(f"DB update_session error: {e}")


def save_message(session_id, role, content):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO conversations (session_id, role, content)
            VALUES (%s, %s, %s)
        """, (session_id, role, content))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB save_message error: {e}")


def get_history(session_id, limit=12):
    try:
        conn = get_connection()
        cur = conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        cur.execute("""
            SELECT role, content FROM conversations
            WHERE session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (session_id, limit))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"role": r["role"], "content": r["content"]}
            for r in rows
        ]
    except Exception as e:
        print(f"DB get_history error: {e}")
        return []


def save_lead_db(
    session_id, name=None, email=None, phone=None,
    business=None, intent=None, stage=None,
    conversation_summary=None, lead_score=None,
    conversation_history=None
):
    try:
        import json
        conn = get_connection()
        cur = conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )

        cur.execute(
            "SELECT id, lead_score FROM leads "
            "WHERE session_id = %s",
            (session_id,)
        )
        existing = cur.fetchone()

        history_json = json.dumps(
            conversation_history or []
        )

        if existing:
            new_score = lead_score or 1
            cur.execute("""
                UPDATE leads SET
                    name = COALESCE(%s, name),
                    email = COALESCE(%s, email),
                    phone = COALESCE(%s, phone),
                    business = COALESCE(%s, business),
                    intent = COALESCE(%s, intent),
                    stage = COALESCE(%s, stage),
                    conversation_summary = COALESCE(
                        %s, conversation_summary
                    ),
                    lead_score = %s,
                    conversation_history = %s,
                    last_updated = NOW()
                WHERE session_id = %s
                RETURNING *
            """, (
                name, email, phone, business,
                intent, stage, conversation_summary,
                new_score, history_json, session_id
            ))
        else:
            cur.execute("""
                INSERT INTO leads (
                    session_id, name, email, phone,
                    business, intent, stage,
                    conversation_summary, lead_score,
                    conversation_history
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING *
            """, (
                session_id, name, email, phone,
                business, intent, stage,
                conversation_summary,
                lead_score or 1, history_json
            ))

        conn.commit()
        result = cur.fetchone()
        cur.close()
        conn.close()
        print(f"✅ Lead saved: {session_id} "
              f"(score: {lead_score})")
        return dict(result) if result else {}

    except Exception as e:
        print(f"DB save_lead error: {e}")
        return {}


def get_all_leads_db():
    try:
        conn = get_connection()
        cur = conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        cur.execute("""
            SELECT * FROM leads
            ORDER BY last_updated DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"DB get_all_leads error: {e}")
        return []


def get_lead_count_db():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM leads")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count
    except Exception as e:
        print(f"DB count error: {e}")
        return 0