from flask import Flask, request, jsonify
from flask_cors import CORS
from core.router import route_query
from core.lead_capture import save_lead, get_all_leads, get_lead_count
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type"])

CALENDLY_LINK = os.getenv(
    "CALENDLY_LINK", "https://calendly.com/your-link"
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DECISION_INTENTS = ["PRICING", "ONBOARDING"]


def is_user_ready_to_book(message, conversation_history):
    """
    Uses GPT to determine if user genuinely wants to book.
    Accepts high OR medium confidence to avoid missing
    clear but contextual confirmations like "yes, now".
    """
    try:
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Agent'}: "
                f"{m['content']}"
                for m in conversation_history[-8:]
            ])

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a booking intent classifier for a sales chatbot.

Your job is to determine if the user wants to book a strategy call
based on the full conversation context.

Read the ENTIRE conversation carefully before deciding.
A short reply like "yes" or "yes, now" or "do it" can be a clear
booking confirmation IF the previous agent message asked about booking.

Think about what the user was responding TO:
- If the agent just asked "want to book a strategy call?" and user says
  "yes" → that IS booking intent, confidence HIGH
- If the agent asked about services and user says "yes" →
  that is agreement, NOT booking, confidence LOW
- "provide the link" or "give me the link" after booking discussion →
  HIGH confidence booking intent
- "do it now" or "let's do it" after booking discussion →
  HIGH confidence booking intent

Return JSON:
{
  "ready": true or false,
  "confidence": "high", "medium", or "low",
  "reason": "one sentence explanation"
}"""
                },
                {
                    "role": "user",
                    "content": f"""Full conversation:
{history_text}

Latest user message: "{message}"

Is this user ready to book right now?"""
                }
            ],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(
            response.choices[0].message.content.strip()
        )

        ready = result.get("ready", False)
        confidence = result.get("confidence", "low")
        reason = result.get("reason", "")

        print(
            f"DEBUG BOOKING CHECK: ready={ready}, "
            f"confidence={confidence}, reason={reason}"
        )

        # Accept high or medium confidence
        # High = explicit booking request
        # Medium = clear contextual confirmation like "yes"
        # after booking CTA
        return ready is True and confidence in ("high", "medium")

    except Exception as e:
        print(f"Booking check error: {e}")
        return False


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    user_name = data.get("name")
    user_email = data.get("email")
    user_phone = data.get("phone")
    user_business = data.get("business")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Get response — returns 8 values
    (response, intent, stage,
     message_count, lead_score,
     conversation_summary,
     conversation_history,
     cta_in_this_response) = route_query(
        user_message, session_id
    )

    # Update session contact flags
    from core.router import session_store
    session = session_store.get(session_id, {})
    if user_email:
        session["email_provided"] = True
    if user_phone:
        session["phone_provided"] = True

    # Read session state for Calendly decision
    cta_previously_shown = session.get("cta_shown", False)

    # Debug — print full state so we can trace issues
    print(
        f"DEBUG CALENDLY STATE: "
        f"stage={stage}, "
        f"message_count={message_count}, "
        f"cta_previously_shown={cta_previously_shown}, "
        f"cta_in_this_response={cta_in_this_response}"
    )

    # Calendly trigger logic
    show_calendly = False

    if stage == "DECISION" and message_count >= 3:
        history = session.get("history", [])

        if cta_in_this_response:
            # CTA appeared in THIS response
            # Don't show Calendly yet — wait for user to respond
            print("DEBUG: CTA shown this turn — waiting for response")

        elif cta_previously_shown:
            # CTA was shown in a previous message
            # User is now responding — check if they want to book
            print("DEBUG: CTA was previously shown — checking booking intent")
            show_calendly = is_user_ready_to_book(
                user_message, history
            )
            if show_calendly:
                print("DEBUG: ✅ Calendly triggered — "
                      "user responded to CTA")

        else:
            # No CTA shown yet but user may be explicitly requesting
            # booking — check anyway for "give me the link" type messages
            print("DEBUG: No CTA yet — checking for explicit booking request")
            show_calendly = is_user_ready_to_book(
                user_message, history
            )
            if show_calendly:
                print("DEBUG: ✅ Calendly triggered — "
                      "explicit booking request detected")

    elif stage != "DECISION":
        print(f"DEBUG: Stage is {stage} — Calendly not checked")
    elif message_count < 3:
        print(f"DEBUG: Message count {message_count} < 3 — "
              f"Calendly not checked yet")

    # Save lead with full conversation history
    if (intent in DECISION_INTENTS or
            stage == "DECISION" or user_email):
        save_lead(
            session_id=session_id,
            name=user_name,
            email=user_email,
            phone=user_phone,
            business=user_business,
            intent=intent,
            stage=stage,
            conversation_summary=conversation_summary,
            lead_score=lead_score,
            conversation_history=conversation_history
        )

    return jsonify({
        "response": response,
        "intent": intent,
        "stage": stage,
        "message_count": message_count,
        "lead_score": lead_score,
        "show_calendly": show_calendly,
        "calendly_link": CALENDLY_LINK if show_calendly else None,
        "session_id": session_id
    })


@app.route("/leads", methods=["GET"])
def leads():
    all_leads = get_all_leads()
    return jsonify({
        "total": len(all_leads),
        "leads": all_leads
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Chanakya is online",
        "leads_captured": get_lead_count()
    })


@app.route("/capture-lead", methods=["POST"])
def capture_lead():
    data = request.json
    session_id = data.get("session_id", "default")

    if not session_id or session_id == "default":
        return jsonify({
            "success": False,
            "message": "Invalid session ID"
        }), 400

    lead = save_lead(
        session_id=session_id,
        name=data.get("name"),
        email=data.get("email"),
        phone=data.get("phone"),
        business=data.get("business"),
        intent="manual_capture",
        stage="captured",
        lead_score=data.get("lead_score", 1)
    )

    return jsonify({
        "success": True,
        "message": "Lead captured successfully",
        "lead": lead
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host="0.0.0.0", port=port)