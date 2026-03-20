from core.intent_layer import detect_intent
from scripts.query import query_rag
from core.database import (
    get_session, update_session,
    save_message, get_history
)
import os


def update_stage(session, intent):
    if intent == "OUT_OF_SCOPE":
        return session["stage"]

    stage = session["stage"]
    count = session.get("meaningful_message_count", 0)

    if stage == "AWARENESS":
        if intent in ["PRICING", "ROI"]:
            session["stage"] = "CONSIDERATION"
        elif intent == "ONBOARDING" and count >= 2:
            session["stage"] = "CONSIDERATION"
        elif intent == "OBJECTION":
            pass
        elif count >= 3 and intent != "OBJECTION":
            session["stage"] = "CONSIDERATION"

    elif stage == "CONSIDERATION":
        if intent == "ONBOARDING":
            session["stage"] = "DECISION"
        elif intent == "PRICING" and count >= 4:
            session["stage"] = "DECISION"
        elif intent == "OBJECTION":
            session["stage"] = "OBJECTION_HANDLING"

    elif stage == "OBJECTION_HANDLING":
        if intent in ["PRICING", "SERVICE", "CASE", "ROI", "FAQ"]:
            session["stage"] = "CONSIDERATION"
        elif intent == "ONBOARDING" and count >= 5:
            session["stage"] = "DECISION"

    elif stage == "DECISION":
        if intent == "OBJECTION":
            session["stage"] = "OBJECTION_HANDLING"

    return session["stage"]


def generate_conversation_summary(history):
    if not history or len(history) < 2:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        conversation_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Agent'}: "
            f"{m['content']}"
            for m in history[-10:]
        ])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Summarize this sales conversation in ONE short sentence.
Include: business type, main interest, buying stage.
Example: "Plumbing business owner asking about Growth Package, ready to book."
Keep under 15 words. Return only the summary."""
                },
                {"role": "user", "content": conversation_text}
            ],
            max_tokens=40,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Summary error: {e}")
        user_messages = [
            m["content"] for m in history
            if m["role"] == "user"
        ]
        return user_messages[-1] if user_messages else None


def calculate_lead_score(session, history):
    try:
        from openai import OpenAI
        import json

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        stage = session.get("stage", "AWARENESS")
        intents = session.get("intents_seen", set())
        if isinstance(intents, str):
            intents = set(intents.split(",")) if intents else set()
        intents = list(intents)
        message_count = session.get("message_count", 0)
        has_email = session.get("email_provided", False)
        has_phone = session.get("phone_provided", False)

        conversation_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Agent'}: "
            f"{m['content']}"
            for m in history[-10:]
        ]) if history else "No conversation yet."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a sales lead quality scorer for a Social Media Marketing Agency.

Score this lead from 1 to 10 based on their buying intent and engagement quality.

Read the FULL conversation carefully and assess holistically.

HIGH score signals (push toward 8-10):
- Asking about specific pricing or packages
- Asking about results, ROI, case studies
- Sharing detailed business context
- Expressing urgency or strong need
- Asking about next steps or getting started
- Providing contact information
- Moving from skeptical to interested

LOW score signals (push toward 1-4):
- Expressing they are not interested
- Saying they already have a solution
- Budget objections with no recovery
- Very short disengaged responses
- Ending the conversation negatively

MEDIUM score signals (5-6):
- Just browsing with mild curiosity
- Asking questions but non-committal

Return ONLY JSON:
{"score": <1-10>, "reason": "<one sentence>"}"""
                },
                {
                    "role": "user",
                    "content": f"""Conversation:
{conversation_text}

Stage: {stage} | Intents: {', '.join(intents) if intents else 'none'}
Messages: {message_count} | Email: {has_email} | Phone: {has_phone}

Score this lead."""
                }
            ],
            max_tokens=60,
            temperature=0,
            response_format={"type": "json_object"}
        )

        import json as j
        result = j.loads(
            response.choices[0].message.content.strip()
        )
        score = max(1, min(10, int(result.get("score", 1))))
        reason = result.get("reason", "")
        print(f"DEBUG LEAD SCORE: {score}/10 — {reason}")
        return score

    except Exception as e:
        print(f"Lead scoring error: {e}")
        fallback = {
            "AWARENESS": 2, "CONSIDERATION": 4,
            "OBJECTION_HANDLING": 3, "DECISION": 7
        }
        return fallback.get(
            session.get("stage", "AWARENESS"), 2
        )


def route_query(user_query, session_id="default"):
    # Get session from Supabase
    session = get_session(session_id)

    # Get conversation history from Supabase
    history = get_history(session_id)

    # Detect intent, knowledge needs, tone
    intent, knowledge_types, tone = detect_intent(
        user_query, history
    )

    # Update tone
    session["tone"] = tone

    # Track intents
    intents_set = session.get("intents_seen", set())
    if isinstance(intents_set, str):
        intents_set = set(
            intents_set.split(",")
        ) if intents_set else set()
    intents_set.add(intent)
    session["intents_seen"] = intents_set

    # Track objections
    if intent == "OBJECTION":
        session["objection_count"] = (
            session.get("objection_count", 0) + 1
        )

    # Increment meaningful count
    if intent != "OUT_OF_SCOPE":
        session["meaningful_message_count"] = (
            session.get("meaningful_message_count", 0) + 1
        )

    # Update stage
    stage = update_stage(session, intent)
    session["stage"] = stage

    # Generate response
    if intent == "OUT_OF_SCOPE":
        response = (
            "I'm here to help grow your business through "
            "GrowthForge Media's proven strategies. "
            "What would you like to know about our "
            "services or results?"
        )
    else:
        primary_filter = None
        if knowledge_types:
            type_priority = [
                "pricing", "case", "services",
                "faq", "onboarding", "agency"
            ]
            for t in type_priority:
                if t in knowledge_types:
                    primary_filter = {"type": t}
                    break

        if not primary_filter:
            intent_filter_map = {
                "SERVICE":    {"type": "services"},
                "PRICING":    {"type": "pricing"},
                "FAQ":        {"type": "faq"},
                "CASE":       {"type": "case"},
                "ONBOARDING": {"type": "onboarding"},
                "ROI":        {"type": "case"},
                "OBJECTION":  {"type": "case"},
                "GENERAL":    {"type": "services"},
            }
            primary_filter = intent_filter_map.get(intent)

        response = query_rag(
            user_query,
            metadata_filter=primary_filter,
            knowledge_types=knowledge_types,
            conversation_history=history,
            session_id=session_id,
            stage=stage,
            message_count=session.get(
                "meaningful_message_count", 0
            ),
            cta_shown=session.get("cta_shown", False),
            tone=tone
        )

    # Track CTA
    cta_in_this_response = (
        "strategy call" in response.lower() and
        "book" in response.lower()
    )
    if cta_in_this_response:
        session["cta_shown"] = True
        session["cta_shown_count"] = (
            session.get("cta_shown_count", 0) + 1
        )

    # Save messages to Supabase
    save_message(session_id, "user", user_query)
    save_message(session_id, "assistant", response)

    # Update message count
    session["message_count"] = (
        session.get("message_count", 0) + 1
    )
    session["last_intent"] = intent

    # Save session to Supabase
    update_session(session_id, {
        "stage": session["stage"],
        "last_intent": intent,
        "message_count": session["message_count"],
        "meaningful_message_count": session.get(
            "meaningful_message_count", 0
        ),
        "cta_shown": session.get("cta_shown", False),
        "cta_shown_count": session.get("cta_shown_count", 0),
        "tone": tone,
        "objection_count": session.get("objection_count", 0),
        "intents_seen": session["intents_seen"]
    })

    # Get updated history for summary and scoring
    updated_history = get_history(session_id)

    # Generate summary and score
    conversation_summary = generate_conversation_summary(
        updated_history
    )
    lead_score = calculate_lead_score(session, updated_history)

    # Update lead score in session
    update_session(session_id, {"lead_score": lead_score})

    return (
        response,
        intent,
        stage,
        session["message_count"],
        lead_score,
        conversation_summary,
        updated_history,
        cta_in_this_response
    )