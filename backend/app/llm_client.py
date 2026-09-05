"""
LLM client abstraction.

Per the project's architecture principle, the LLM never touches anything
sensitive (payments, authorization) directly - here it only has ONE job:
turn a natural-language shopping message into structured JSON the rest of
the system can trust the *shape* of (not the truth of - see note below).

This module is intentionally the only place that knows how to talk to a
specific model provider. `OllamaLLMClient` is the local implementation for
now; a future `AnthropicLLMClient` / `OpenAILLMClient` can implement the
same `LLMClient` interface and be swapped in via config without touching
the agent/route logic.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .config import settings
from .merchant_tools import KNOWN_CATEGORIES
from .schemas import ChatMessage, ShoppingContext

logger = logging.getLogger("agent_buyer.llm")

# Bugfix round 15: "recommendation" removed - Semi-Autonomous already did
# everything that mode did (search, reason, show candidates) plus buy-on-
# approval, so it was a strict subset with no purpose of its own.
VALID_MODES = {"semi_autonomous", "fully_autonomous"}


@dataclass
class UnderstandResult:
    reply: str
    intent: ShoppingContext
    # V2 Batch 2: two extra signals extracted in the SAME LLM call, so the
    # frontend can (a) auto-trigger a search instead of requiring a "Find
    # products" click, and (b) infer the autonomy mode from how the user
    # phrased the request instead of requiring a mode-button click. Both
    # are proposals from the LLM, not decisions with any spending power -
    # a mode switch alone can never authorize a payment (Guardian still
    # requires a human-granted delegated authority for fully_autonomous
    # and an explicit per-order approval for semi_autonomous), so a wrong
    # guess here is a UI annoyance, never a security hole.
    ready_to_search: bool = False
    desired_mode: Optional[str] = None
    # Bugfix round 8: instead of only trusting a fixed keyword-regex gate to
    # decide whether a proposed desired_mode is legitimate, the LLM must now
    # justify its proposal with an exact quote from the user's own message.
    # The backend verifies that quote actually appears in the raw message
    # (a cheap, deterministic "grounding" check) before honoring the mode
    # switch - this lets genuinely new phrasing work (versatility) while
    # still refusing anything the model can't actually point to (safety).
    mode_evidence: Optional[str] = None

_SYSTEM_PROMPT_TEMPLATE = """You are the natural-language understanding layer for a shopping \
assistant called Agent Buyer.

Your ONLY job on every turn is to:
1. Read the user's latest message (and the conversation so far).
2. Extract any shopping intent mentioned in THIS message into structured JSON.
3. Write a short, natural, friendly reply. Do not invent product names, prices, \
or availability - you have no access to real products yet, so never claim to have \
found or recommended anything.

Rules for clarifying questions - be economical, users disengage if you interrogate \
them one detail at a time:
- A search needs at least a CATEGORY and a BUDGET to be worth running - always treat \
both as required before you're ready, never just the category. If either (or both) \
is missing, ask for them TOGETHER in ONE short combined message (e.g. "What kind of \
product are you looking for, and what's your budget?", or if category is already \
known, "What's your budget for that?") - never ask about category alone in one turn \
and then budget in a separate later turn. A key attribute (brand, use \
case, color, etc.) is a nice-to-have you can fold into the same combined question if \
relevant, but is never itself a substitute for budget.
- You only get ONE such round of clarifying questions per conversation. If you have \
already asked a clarifying question earlier in this conversation and the user has \
replied to it (even briefly, even with just one word or "yes"), do NOT ask another \
clarifying question this turn - the user has already engaged once, asking again \
reads as not listening. Work with whatever you now have, even if budget is still \
missing after that one round, and write a reply that reads as moving forward (e.g. \
"Got it - searching for X now.") rather than asking again.
- A message titled "Already confirmed from earlier in this conversation" may appear \
right before the user's latest message, listing category/budget/keywords/attributes \
already gathered. Treat everything in it as settled fact - NEVER ask the user about \
anything it lists (a shoe size, a color, a brand, budget, category, or anything else), \
even if this specific message doesn't repeat it. Only ask about something genuinely \
missing from BOTH this message AND that list.

Respond with STRICT JSON only, matching exactly this shape, no markdown fences, \
no extra commentary outside the JSON:

{{
  "reply": "<your natural language reply to the user>",
  "intent": {{
    "category": "<string or null>",
    "keywords": ["<string>", "..."],
    "min_price": <number or null>,
    "max_price": <number or null>,
    "currency": "<3-letter code, default INR>",
    "attributes": {{"<key>": "<value>", "...": "..."}},
    "notes": "<string or null>"
  }},
  "ready_to_search": <true or false>,
  "desired_mode": "<'semi_autonomous' | 'fully_autonomous' | null>",
  "mode_evidence": "<exact substring copied verbatim from the user's message that justifies desired_mode, or null>"
}}

Rules for "intent":
- Only include fields the user actually stated or clearly implied THIS message.
- Leave a field null / empty if not mentioned in this message (do not repeat \
old values - the backend merges turns itself).
- "attributes" is for anything that doesn't fit category/price, e.g. \
use_case, size, color, brand, material, gender.
- Numbers only for min_price/max_price (no currency symbols, no commas).
- Budget interpretation is important: a plain statement like "my budget is 2000", \
"under 2000", "around 2000", or "budget would be 2000" means max_price=2000. \
Only set min_price when the user explicitly states a LOWER bound, using words like \
"at least", "minimum", "more than", or "starting from". When the user restates or \
lowers their budget (e.g. "actually make it 3000"), that is a new max_price, not a \
min_price, unless they say otherwise.
- "category" MUST be one of exactly these catalog categories (or null if none fit): \
{categories}. Map the user's own words to the closest one of these - e.g. "sports \
watch"/"fitness watch" -> "smartwatches", "sneakers" -> "casual sneakers", "office \
shoes" -> "formal shoes", "bag"/"rucksack" -> "backpacks", "buds"/"headphones" -> \
"wireless earbuds". Never invent a category outside this list.

Rules for "ready_to_search" (this replaces a manual "Find products" click, so get \
this right):
- true ONLY when you now have BOTH a category AND a budget (min_price or max_price) \
AND your "reply" is NOT asking another clarifying question - it should instead be a \
short confirmation that you're searching now (e.g. "Got it - searching for running \
shoes under 4000 now."). Category alone is NOT enough, even if it's the only thing \
you're missing - always get a budget too before the first search, per the combined \
clarifying-question rule above.
- true whenever the user explicitly tells you to search/proceed/go ahead now (e.g. \
"search now", "just find something", "go ahead", "any budget is fine"), even if \
budget is still unknown - respect an explicit instruction to stop asking and search.
- false whenever your "reply" is itself asking the user something you still need to \
know before a search would be useful (missing category and/or budget, or the message \
is too vague to search at all) - but remember the one-clarifying-round rule above: if \
you already asked once this conversation (for category and/or budget together) and \
the user replied, don't ask a second time even if budget is still missing - set this \
true and move forward with whatever you have.
- IMPORTANT: if a single message already gives you a category PLUS a budget or a key \
attribute AND also gives purchase/autonomy intent (e.g. "Find me running shoes under \
5000 that are lightweight and comfortable. I want you to buy whatever you think is \
best.") - that message is immediately ready to search. Do NOT ask a confirming \
question like "should I search now?" or "shall I go ahead?" in this case - your \
"reply" should just confirm you're searching now, and "ready_to_search" must be true \
on this SAME turn. Never make the user say "okay go ahead" as a separate turn when \
they already gave you everything needed in one message.

Rules for "desired_mode" (this replaces manually clicking a mode button, so be \
conservative - a wrong guess here is a UI-only annoyance since no purchase can \
actually happen from a mode switch alone, but guess only when the signal is clear):
- "fully_autonomous": the user says things like "you decide and buy it", "just handle \
it end to end", "buy the best one for me automatically", "I trust you, go ahead and \
purchase it" - explicit authorization for the agent to both choose AND pay without \
asking again. This ALSO includes short, bare purchase imperatives directed at a \
product already discussed or shown in this conversation - e.g. "buy it", "okay buy \
it", "go ahead and buy it", "yes, get it", "just buy it already", "purchase it now". \
A short imperative like this, especially right after a product has been surfaced, \
means the user wants the purchase to happen immediately without a further approval \
step - treat it as fully_autonomous, NOT semi_autonomous, even though it's brief.
- "semi_autonomous": the user says things like "show me some options and I'll pick", \
"let me approve before you buy", "recommend something but check with me first", \
"show me the product(s) first before buying", "show me what you found before you \
purchase anything", "let me see it before you buy" - they want to see/review \
candidates or a specific product BEFORE any purchase happens and want to explicitly \
approve that specific purchase - they are open to the agent proceeding to a purchase \
flow once they approve, but not before. Any phrasing that asks to see/review \
something "before" a purchase, or to approve first, is semi_autonomous - this is a \
common and important signal, do not miss it. This ALSO covers phrasing that rules out \
a purchase happening automatically at all - e.g. "just show me options", "I'm only \
browsing", "don't buy anything, just suggest" - there is no lower/browse-only mode \
below this one, so ruling out an automatic purchase means semi_autonomous (search, \
reason, and show candidates, but never buy without explicit approval), not null.
- null: the message gives NO clear signal about autonomy/purchasing intent at all \
(e.g. it's purely describing what product they want). Do not guess from vague or \
generic language - null is the safe default and should be the most common value.

Rules for "mode_evidence" (REQUIRED whenever "desired_mode" is not null):
- Copy the EXACT words from the user's message (verbatim, same casing/punctuation, no \
paraphrasing, no summarizing) that most directly justify the "desired_mode" you chose. \
Keep it short - just the phrase that carries the signal, e.g. "buy it", "let me \
approve first", "go ahead and purchase it" - not the whole message.
- CRITICAL: "mode_evidence" must be copied from the USER's message, never from your \
own "reply" field above it. Do NOT describe or summarize what you're about to do \
(e.g. do NOT write "you can review and approve before we proceed to purchase" - that's \
your own reply text, not something the user said) - copy the user's actual words, even \
if they're short, informal, or contain typos (e.g. the user wrote "I want choose,review \
and buy" - the correct evidence is "want choose,review" or "review", copied exactly as \
they typed it, NOT a cleaned-up paraphrase of your own answer).
- The backend will check that this exact text actually appears in the user's message. \
If you cannot find real words in the message that justify your answer, that is a sign \
you should not have proposed a mode change at all - set both "desired_mode" and \
"mode_evidence" to null instead of inventing a quote.
- If a message has BOTH a purchase signal (e.g. "buy") AND a review/approval signal \
(e.g. "let me choose", "and pick", "I'll decide") in the same sentence, prefer \
"semi_autonomous" and quote the review/approval phrase - a user asking to buy AND \
choose/pick/decide wants to review before the purchase completes, which is what \
semi_autonomous means, even though the word "buy" is also present.
- Always null when "desired_mode" is null.

A few worked examples (message -> desired_mode, mode_evidence):
- "Okay buy it" (a candidate product already shown earlier in the conversation) -> \
"fully_autonomous", "buy it"
- "Find me running shoes under 5000 that are lightweight. Buy whatever you think is \
best." -> "fully_autonomous", "Buy whatever you think is best"
- "Show the products first before buying" -> "semi_autonomous", "before buying"
- "Show me some options and I'll pick one to buy" -> "semi_autonomous", "I'll pick"
- "I want to buy and choose those products" -> "semi_autonomous", "and choose"
- "I want choose,review  and buy" -> "semi_autonomous", "want choose,review" (copied from \
the user's own words, typos and all - NOT "you can review and approve before we \
proceed to purchase", which would be your own reply text, not the user's)
- "I want running shoes under 5000" (no purchase/approval signal at all) -> null, null
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(categories=", ".join(KNOWN_CATEGORIES))


def _summarize_known_context(context: Optional[ShoppingContext]) -> str:
    """Bugfix round 9 (issue j, piece 1). Renders only the non-empty parts
    of an accumulated ShoppingContext as a short summary for the prompt,
    or "" if nothing is known yet (so the caller can skip the whole block
    rather than sending an empty/noisy one). Deliberately field-agnostic -
    it reports whatever is actually present, including free-form
    attributes like shoe size or color, not just category/budget - so this
    doesn't need updating every time a new attribute type shows up."""
    if context is None:
        return ""
    lines: list[str] = []
    if context.category:
        lines.append(f"- category: {context.category}")
    if context.min_price is not None or context.max_price is not None:
        lo = context.min_price if context.min_price is not None else "any"
        hi = context.max_price if context.max_price is not None else "any"
        lines.append(f"- budget: {lo} to {hi} {context.currency}")
    if context.keywords:
        lines.append(f"- keywords: {', '.join(context.keywords)}")
    if context.attributes:
        attr_str = ", ".join(f"{k}: {v}" for k, v in context.attributes.items())
        lines.append(f"- attributes: {attr_str}")
    if context.notes:
        lines.append(f"- notes: {context.notes}")
    return "\n".join(lines)


class LLMClient(ABC):
    @abstractmethod
    async def understand(
        self,
        message: str,
        history: list[ChatMessage],
        accumulated_context: Optional[ShoppingContext] = None,
    ) -> UnderstandResult:
        """Return the reply, extracted turn intent, and V2 Batch 2's
        auto-search / mode-inference signals for this turn.

        `accumulated_context` (bugfix round 9, issue j/piece 1) is the
        backend's own merged ShoppingContext from BEFORE this turn - what's
        already confirmed from earlier in the conversation (category,
        budget, every attribute). Previously this method only ever saw the
        raw conversation history and had to re-derive what's already known
        by re-reading old messages; passing the actual merged state
        directly lets the prompt tell the model "don't ask about anything
        already listed here" for ANY field, not just the two the backend
        can independently double-check (category/budget - see
        `_apply_single_budget_guard`/`_apply_category_fallback` in
        main.py). Optional/defaulted so a caller that doesn't have it yet
        still works."""
        raise NotImplementedError

    @abstractmethod
    async def raw_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_format: bool = False,
    ) -> dict:
        """Lower-level chat call for callers (like the recommendation
        agent) that need tool-calling instead of the fixed understand()
        contract. Returns a raw {role, content, tool_calls?} message dict."""
        raise NotImplementedError


class OllamaLLMClient(LLMClient):
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model

    async def understand(
        self,
        message: str,
        history: list[ChatMessage],
        accumulated_context: Optional[ShoppingContext] = None,
    ) -> UnderstandResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})
        # Bugfix round 9 (issue j, piece 1): hand the backend's own merged
        # context to the model as ground truth, right before the latest
        # message - see the docstring on the abstract method above for why.
        # Only included when there's actually something to report, so an
        # empty/fresh session doesn't add noise to the prompt.
        context_summary = _summarize_known_context(accumulated_context)
        if context_summary:
            messages.append({
                "role": "system",
                "content": (
                    "Already confirmed from earlier in this conversation - do NOT ask "
                    "the user about anything listed below, whether or not this "
                    "message repeats it. Only ask about what's still genuinely "
                    "missing.\n" + context_summary
                ),
            })
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }

        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running "
                f"and is model '{self.model}' pulled (`ollama pull {self.model}`)?"
            ) from exc

        data = resp.json()
        raw_content = data.get("message", {}).get("content", "")
        return self._parse(raw_content)

    async def raw_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_format: bool = False,
    ) -> dict:
        """Lower-level chat call used by the recommendation agent, which
        needs tool-calling rather than the fixed understand() JSON shape.
        Returns the raw `message` dict from Ollama's response
        (role/content/tool_calls)."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if tools:
            payload["tools"] = tools
        if json_format:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                resp = await client.post(f"{self.host}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running "
                f"and is model '{self.model}' pulled (`ollama pull {self.model}`)?"
            ) from exc

        data = resp.json()
        return data.get("message", {})

    def _parse(self, raw_content: str) -> UnderstandResult:
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON content, falling back. Raw: %r", raw_content)
            return UnderstandResult(
                reply=raw_content.strip() or "Sorry, could you rephrase that?",
                intent=ShoppingContext(),
            )

        reply = parsed.get("reply") or "Got it."
        intent_raw = parsed.get("intent") or {}

        # Be defensive - a local model may return slightly malformed shapes
        # (e.g. a string instead of a list for keywords). Never let a bad
        # LLM response crash the request.
        try:
            keywords = intent_raw.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]

            attributes = intent_raw.get("attributes") or {}
            if not isinstance(attributes, dict):
                attributes = {}

            intent = ShoppingContext(
                category=intent_raw.get("category") or None,
                keywords=[str(k) for k in keywords if k],
                min_price=_to_float(intent_raw.get("min_price")),
                max_price=_to_float(intent_raw.get("max_price")),
                currency=intent_raw.get("currency") or "INR",
                attributes={str(k): str(v) for k, v in attributes.items() if v is not None},
                notes=intent_raw.get("notes") or None,
            )
        except Exception:
            logger.exception("Failed to coerce LLM intent into ShoppingContext: %r", intent_raw)
            intent = ShoppingContext()

        ready_to_search = bool(parsed.get("ready_to_search") is True)

        desired_mode_raw = parsed.get("desired_mode")
        desired_mode: Optional[str] = None
        if isinstance(desired_mode_raw, str) and desired_mode_raw.strip().lower() in VALID_MODES:
            desired_mode = desired_mode_raw.strip().lower()
        elif desired_mode_raw not in (None, "", "null"):
            # The model returned something that isn't one of the three known
            # modes (typo, hallucinated value, etc.) - never let a garbage
            # value reach the session's mode; treat it the same as "no
            # signal" rather than guessing which mode was meant.
            logger.warning("Ignoring unrecognized desired_mode from LLM: %r", desired_mode_raw)

        mode_evidence_raw = parsed.get("mode_evidence")
        mode_evidence: Optional[str] = (
            mode_evidence_raw.strip()
            if isinstance(mode_evidence_raw, str) and mode_evidence_raw.strip()
            else None
        )

        return UnderstandResult(
            reply=reply,
            intent=intent,
            ready_to_search=ready_to_search,
            desired_mode=desired_mode,
            mode_evidence=mode_evidence,
        )


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_llm_client() -> LLMClient:
    # Swap point: return AnthropicLLMClient()/OpenAILLMClient() here later,
    # selected by a config value, without touching main.py.
    return OllamaLLMClient()
