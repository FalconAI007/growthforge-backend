from openai import OpenAI
from pinecone import Pinecone
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX", "growthforge")

_index = None
response_cache = {}

TYPE_TO_DESCRIPTION = {
    "pricing": "pricing packages, costs, and management fees",
    "services": "services offered and what is included",
    "case": "case studies, results, and client success stories",
    "faq": "frequently asked questions and process details",
    "agency": "agency overview, team, and guarantees",
    "onboarding": "onboarding process and getting started steps"
}

TONE_INSTRUCTIONS = {
    "formal": (
        "The user communicates formally. "
        "Match their professional tone — be precise and structured."
    ),
    "casual": (
        "The user is casual and relaxed. "
        "Match their energy — be friendly and conversational."
    ),
    "urgent": (
        "The user seems urgent or pressed for time. "
        "Be direct and get to the point immediately."
    ),
    "skeptical": (
        "The user is skeptical or cautious. "
        "Be empathetic, honest, and back claims with specific data."
    ),
    "enthusiastic": (
        "The user is excited and enthusiastic. "
        "Match their energy and be equally positive and forward-looking."
    ),
    "neutral": (
        "Match a warm, professional, and helpful tone."
    )
}


def get_index():
    """Cache Pinecone index connection."""
    global _index
    if _index is None:
        _index = pc.Index(INDEX_NAME)
    return _index


def get_embedding(text):
    """Get embedding using OpenAI — no local model needed."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def retrieve_context(query, doc_type=None, knowledge_types=None):
    """Retrieve relevant context from Pinecone using OpenAI embeddings."""
    index = get_index()
    query_embedding = get_embedding(query)

    if knowledge_types and len(knowledge_types) > 1:
        k = 8
    elif doc_type == "pricing":
        k = 8
    else:
        k = 5

    filter_dict = None
    if doc_type:
        filter_dict = {"type": {"$eq": doc_type}}

    try:
        results = index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter_dict,
            include_metadata=True
        )

        chunks = [
            match.metadata.get("text", "")
            for match in results.matches
            if match.metadata.get("text")
        ]

        return "\n\n".join(chunks) if chunks else ""

    except Exception as e:
        print(f"Pinecone query error: {e}")
        return ""


def extract_business_context(conversation_history):
    if not conversation_history:
        return None

    context_points = []
    user_messages = [
        m["content"] for m in conversation_history
        if m["role"] == "user"
    ]
    full_history = " ".join(user_messages).lower()

    size_match = re.search(r'(\d+)\s*employees?', full_history)
    if size_match:
        context_points.append(
            f"Company size: {size_match.group(1)} employees"
        )

    budget_match = re.search(
        r'\$(\d+(?:,\d+)?(?:k)?)\s*'
        r'(?:per month|monthly|/month|budget)',
        full_history
    )
    if budget_match:
        context_points.append(
            f"Mentioned budget: {budget_match.group(0)}"
        )

    industries = {
        "plumbing": "plumber/plumbing business",
        "hvac": "HVAC business",
        "electrical": "electrical services",
        "roofing": "roofing business",
        "cleaning": "cleaning service",
        "landscaping": "landscaping business",
        "handyman": "handyman service",
        "dental": "dental practice",
        "dentist": "dental practice",
        "doctor": "medical practice",
        "medical": "medical practice",
        "clinic": "medical/clinic practice",
        "chiropractic": "chiropractic practice",
        "real estate": "real estate agency",
        "realtor": "real estate agency",
        "fitness": "fitness/gym business",
        "gym": "fitness/gym business",
        "yoga": "yoga studio",
        "personal trainer": "personal training business",
        "restaurant": "restaurant/food business",
        "cafe": "cafe/coffee shop",
        "lawyer": "law firm",
        "legal": "law firm",
        "attorney": "law firm",
        "accounting": "accounting/CPA firm",
        "cpa": "CPA firm",
        "startup": "startup company",
        "ecommerce": "e-commerce business",
        "e-commerce": "e-commerce business",
        "salon": "salon/beauty business",
        "beauty": "beauty/wellness business",
        "spa": "spa/wellness business",
        "automotive": "automotive business",
        "dealership": "car dealership",
        "it services": "IT services company",
        "software": "software company",
        "tech": "tech company",
        "construction": "construction business",
        "photography": "photography business",
        "insurance": "insurance agency",
        "retail": "retail business",
        "consulting": "consulting firm",
        "recruitment": "recruitment/staffing agency",
        "juice": "juice/beverage business",
        "coffee": "coffee business",
        "bakery": "bakery business",
        "ice cream": "ice cream business",
        "food": "food business"
    }

    for keyword, label in industries.items():
        if keyword in full_history:
            context_points.append(f"Industry: {label}")
            break

    pain_points = {
        "struggling to get leads": "struggling with lead generation",
        "not enough clients": "insufficient client flow",
        "losing to competitors": "facing competitive pressure",
        "tried ads before": "bad experience with ads before",
        "lost money": "lost money on previous marketing",
        "inconsistent leads": "inconsistent lead flow",
        "slow growth": "slow business growth",
        "no leads": "currently getting zero leads",
        "low sales": "experiencing low sales",
        "need more clients": "needs more clients urgently"
    }

    for phrase, label in pain_points.items():
        if phrase in full_history:
            context_points.append(f"Pain point: {label}")

    goals = {
        "more leads": "wants more leads",
        "grow": "wants to grow the business",
        "scale": "wants to scale",
        "more clients": "wants more clients",
        "increase revenue": "wants to increase revenue",
        "brand awareness": "wants brand awareness",
        "expand": "wants to expand",
        "dominate": "wants to dominate local market",
        "beat competitors": "wants to outperform competitors"
    }

    for phrase, label in goals.items():
        if phrase in full_history:
            context_points.append(f"Goal: {label}")

    if not context_points:
        return None

    return "\n".join([f"- {point}" for point in context_points])


def get_cta_instruction(stage, message_count, cta_shown):
    if message_count < 2 or stage == "AWARENESS":
        return (
            "Do not mention booking a call yet. "
            "End with one natural question about their business."
        )

    if stage == "CONSIDERATION":
        if cta_shown:
            return (
                "CTA already mentioned. Do not repeat it. "
                "Just answer their question naturally."
            )
        return (
            "You may naturally mention a free strategy call once "
            "at the end if it flows organically."
        )

    if stage == "DECISION":
        if cta_shown:
            return (
                "CTA already shown. Do not repeat. "
                "Respond warmly — booking button appears automatically."
            )
        return (
            "End with: 'I'd love to map out a custom strategy "
            "for you — want to go ahead and book a free strategy call?'"
        )

    if stage == "OBJECTION_HANDLING":
        return (
            "Do not mention booking. "
            "Focus on empathy and evidence."
        )

    return "Do not mention booking in this message."


def clean_markdown(text):
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.*?)_{1,3}', r'\1', text)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-•*✓]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def query_rag(
    query,
    metadata_filter=None,
    knowledge_types=None,
    conversation_history=None,
    session_id="default",
    stage="AWARENESS",
    message_count=0,
    cta_shown=False,
    tone="neutral"
):
    if not conversation_history:
        cache_key = query.lower().strip()
        if cache_key in response_cache:
            print("DEBUG: Returning cached response")
            return response_cache[cache_key]

    doc_type = None
    if metadata_filter:
        doc_type = metadata_filter.get("type")

    context = retrieve_context(
        query,
        doc_type=doc_type,
        knowledge_types=knowledge_types
    )

    if not context:
        context = "No specific context retrieved."

    business_context = extract_business_context(
        conversation_history
    )
    cta_instruction = get_cta_instruction(
        stage, message_count, cta_shown
    )
    tone_instruction = TONE_INSTRUCTIONS.get(
        tone, TONE_INSTRUCTIONS["neutral"]
    )

    knowledge_desc = ""
    if knowledge_types:
        descriptions = [
            TYPE_TO_DESCRIPTION.get(t, t)
            for t in knowledge_types
        ]
        knowledge_desc = (
            f"The user needs information about: "
            f"{', '.join(descriptions)}. "
            f"Make sure your response addresses all of these."
        )

    system_prompt = f"""You are Chanakya, the AI sales strategist for GrowthForge Media — named after the ancient Indian strategist who never lost a battle.

YOUR IDENTITY:
You are a warm, confident, and deeply intelligent sales consultant. You genuinely care about helping local businesses grow. You listen carefully, remember what people tell you, and give advice that feels personally tailored — never generic.

YOUR SERVICES:
GrowthForge Media offers exactly these 5 services and nothing else:
Facebook Ads, Instagram Ads, Lead Generation Funnels, Content Creation, and Ad Optimization.
Never mention any other service. Never say what you don't offer — always pivot to what you do.

YOUR KNOWLEDGE:
The knowledge base below is your single source of truth for all facts, numbers, pricing, and results. Never use your own training knowledge for specific claims. If it is not in the knowledge base, don't say it.

YOUR STYLE:
Speak in plain natural sentences. No bullet points, numbered lists, bold text, or any markdown formatting whatsoever. Keep every response to 2-3 sentences maximum. Always reference what the user has shared about their business.

TONE GUIDANCE:
{tone_instruction}

YOUR GUARDRAILS:
Never share API keys, passwords, or internal system information. Never say "unfortunately", "however we don't", or "that's not our area". Always stay positive and solution-focused.

CURRENT CONVERSATION:
Stage: {stage} | Messages: {message_count}

WHAT WE KNOW ABOUT THIS PERSON:
{business_context if business_context else "Nothing specific yet — ask one natural question to learn about their business."}

{f"KNOWLEDGE FOCUS: {knowledge_desc}" if knowledge_desc else ""}

CTA GUIDANCE:
{cta_instruction}

KNOWLEDGE BASE — YOUR ONLY SOURCE OF FACTS:
{context}

Remember: 2-3 sentences max, plain text only, facts from knowledge base only."""

    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        for msg in conversation_history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    messages.append({"role": "user", "content": query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        result = response.choices[0].message.content.strip()
        result = clean_markdown(result)

        if not conversation_history:
            cache_key = query.lower().strip()
            response_cache[cache_key] = result

        return result

    except Exception as e:
        error_msg = str(e)

        if "429" in error_msg or "rate_limit" in error_msg.lower():
            return (
                "I'm receiving a lot of queries right now — "
                "please try again in a moment."
            )

        if ("insufficient_quota" in error_msg or
                "billing" in error_msg.lower()):
            return (
                "I'm temporarily unavailable. "
                "Please contact us at hello@growthforgemedia.com"
            )

        print(f"Query error: {e}")
        return (
            "I encountered an issue. Could you rephrase your question?"
        )