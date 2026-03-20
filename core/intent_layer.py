from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_intent(query, conversation_history=None):
    """
    Fully GPT-driven intent detection.
    No hardcoded keywords or examples — pure reasoning.
    Returns: intent, knowledge_types, tone
    """
    try:
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{'User' if m['role'] == 'user' else 'Agent'}: "
                f"{m['content']}"
                for m in conversation_history[-6:]
            ])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a conversation intent analyzer for a Social Media Marketing Agency sales chatbot.

Your task is to deeply understand the TRUE meaning and intent behind a user's message in the context of a sales conversation about marketing services.

CONTEXT:
This is a sales chatbot for a marketing agency that helps local businesses grow through social media advertising. Every conversation should ultimately be about helping the user grow their business.

REASONING FRAMEWORK:
Before classifying, think about these dimensions:

DIMENSION 1 — What is the user's relationship to the conversation?
Are they sharing something about themselves and their situation?
Or are they actively seeking something from the agent?
Or are they reacting to what was said?

DIMENSION 2 — What is the user's emotional and intentional state?
Are they open and curious? Resistant or skeptical? Ready to act?
Disengaged or testing boundaries? Exploring casually?

DIMENSION 3 — Is this message relevant to business growth?
Does this message connect to the user's business, their marketing needs,
or the agency's ability to help them?
Or is this a general knowledge question, entertainment, or off-topic entirely?
A message is off-topic if a knowledgeable sales agent would have no
relevant business-focused response to it.

DIMENSION 4 — What does the user need next?
What would genuinely serve this user right now in their buying journey?

INTENTS — choose the single most accurate one:

GENERAL: User is expressing, sharing, or greeting — not yet asking
PRICING: User wants to understand costs or packages
SERVICE: User wants to know what the agency can do for them
CASE: User wants proof, results, or social validation
ROI: User wants to understand financial return on investment
FAQ: User wants to understand process, timeline, or operations
ONBOARDING: User is actively ready to start or book right now
OBJECTION: User is resistant, skeptical, or pushing back
OUT_OF_SCOPE: Message has no connection to the user's business growth
              or the agency's services — a knowledgeable sales agent
              would have nothing relevant to say about it in a
              business context

KNOWLEDGE TYPES — what data would make the best response:
Choose from: pricing, services, case, faq, agency, onboarding
Pick what genuinely serves the user's current need.
Return empty array for OUT_OF_SCOPE.

TONE — the user's communication style:
formal, casual, urgent, skeptical, enthusiastic, neutral

Return ONLY valid JSON:
{
  "intent": "INTENT_HERE",
  "knowledge_types": ["type1"],
  "tone": "tone_here"
}"""
                },
                {
                    "role": "user",
                    "content": f"""Conversation so far:
{history_text if history_text else "This is the first message."}

Current message: "{query}"

Think through the 4 dimensions carefully then return JSON."""
                }
            ],
            max_tokens=120,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(
            response.choices[0].message.content.strip()
        )

        intent = result.get("intent", "GENERAL").upper()
        knowledge_types = result.get("knowledge_types", [])
        tone = result.get("tone", "neutral")

        valid_intents = [
            "PRICING", "SERVICE", "CASE", "ROI",
            "ONBOARDING", "FAQ", "OBJECTION",
            "GENERAL", "OUT_OF_SCOPE"
        ]

        if intent not in valid_intents:
            intent = "GENERAL"

        print(
            f"DEBUG INTENT: {intent} | "
            f"KNOWLEDGE: {knowledge_types} | "
            f"TONE: {tone}"
        )

        return intent, knowledge_types, tone

    except Exception as e:
        print(f"Intent detection error: {e}")
        return "GENERAL", ["services"], "neutral"