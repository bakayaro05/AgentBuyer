"""
Agent Buyer - backend entrypoint (Phase 1).

Scope of this phase, deliberately: understand the user's natural-language
shopping request and let you verify, turn by turn, that the extracted
context is correct. No product search, no merchants, no payments yet -
those come in later phases per the project's development order.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .agent import RecommendationResult, run_recommendation, run_recommendation_iter
from .config import settings
from .database import get_session as get_db_session
from .guardian import GuardianResult, evaluate as guardian_evaluate, evaluate_iter as guardian_evaluate_iter
from .llm_client import get_llm_client
from .merchant_tools import KNOWN_CATEGORIES, get_product_by_sku, search_products
from .models import DelegatedAuthority, Order, Product
from .payment_gateway import get_payment_gateway
from .schemas import (
    ApproveRequest,
    AuditEntryOut,
    AuthorityOut,
    AuthorityProposalOut,
    CheckoutRequest,
    CheckoutResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GrantAuthorityRequest,
    GuardianCheckOut,
    GuardianResultOut,
    OrderEventOut,
    OrderOut,
    PaymentConfigOut,
    ProductOut,
    RecommendRequest,
    RecommendResponse,
    ResetRequest,
    RevokeAuthorityRequest,
    SelectRequest,
    SetModeRequest,
    ShoppingContext,
    SimulatePaymentRequest,
    PaymentAttemptFailedRequest,
    VerifyPaymentRequest,
)
from .seed import init_and_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_buyer")

app = FastAPI(title="Agent Buyer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = get_llm_client()

init_and_seed()

# In-memory session store. Fine for a single-user local demo; swap for
# Redis/Postgres-backed sessions later if this needs to survive restarts
# or serve multiple concurrent users.
_sessions: dict[str, dict] = {}


# --- V2 "thought stream" SSE plumbing -------------------------------------
#
# The frontend's live, animated thinking view (spinner + one-liner steps,
# click-to-expand) needs each backend step pushed the moment it actually
# happens - not replayed after a JSON response comes back. Every streamed
# endpoint below emits `text/event-stream` frames shaped either
# {"type": "step", "id": ..., "label": ..., "detail": ..., "status": ...}
# or {"type": "result", ...the same payload the non-streaming endpoint
# would have returned...} or {"type": "error", "message": ...}.
#
# STEP_PACE_SECONDS is a small floor on how fast consecutive steps can
# appear - genuine backend steps still drive the sequence and its
# content, this only keeps two near-instant steps (e.g. two Guardian
# checks that both short-circuit in microseconds) from flashing past
# unreadably. It is not a replacement for real progress.
STEP_PACE_SECONDS = 0.3


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# Bugfix round 4 follow-up: the LLM's `desired_mode` field has been caught
# switching straight to fully_autonomous on a message that never actually
# said anything about buying - just product criteria (category, budget,
# attributes) with no purchase-imperative wording at all (e.g. "Find me a
# good running shoes under 5000rs thats lightweight and comfortable" - no
# "buy" anywhere). The worked examples in llm_client.py's prompt pair
# "lots of product detail in one message" with "buy whatever you think is
# best" in the SAME example, and the local 8B model appears to sometimes
# pattern-match on the former alone and hallucinate the purchase intent
# rather than actually requiring it - the same class of failure this
# codebase has repeatedly found (invented search filters, hallucinated
# products, misread budgets). Per the established "never fully trust the
# LLM's own JSON" pattern, a mode switch INTO fully_autonomous or
# semi_autonomous now additionally requires the raw user message to
# contain wording that actually supports it, checked here in the backend
# rather than trusted at face value - a switch with no supporting wording
# is dropped (logged) exactly like an unrecognized category or a stale
# min_price elsewhere in this file. (Bugfix round 15: this comment used to
# also note that "recommendation" needed no such guard, since it only
# narrowed capability - that mode is gone now, see `_apply_desired_mode`'s
# docstring for why.)
#
# Bugfix round 8: this fixed keyword list caught the round 5 bug, but it
# isn't versatile - phrasing nobody enumerated here (e.g. "I want to buy
# and choose those products", where "and choose" signals the user wants to
# review before buying) still falls through to whichever single-signal
# list happens to match, ignoring the more nuanced intent. `_apply_desired_mode`
# now checks LLM-quoted evidence ("mode_evidence") against the raw message
# FIRST (see `_mode_evidence_is_grounded`) - this handles any phrasing the
# LLM can correctly point to in the user's own words, not just what's
# enumerated below. These two regexes are kept as a deterministic FALLBACK,
# used only when the LLM's quote isn't grounded in the message (empty,
# paraphrased, or hallucinated) - so a correct classification with a
# botched quote still goes through, and the original round 5 protection
# (nothing gets through with zero supporting evidence of either kind)
# still holds.
_FULLY_AUTONOMOUS_SIGNAL = re.compile(
    r"\bbuy\b|\bpurchase\b|\bpurchasing\b|\border it\b|\bget it\b|"
    r"go ahead and (buy|purchase|get|order)|you decide|i trust you|"
    r"handle it (end to end|for me|yourself)?|just (buy|get|purchase|order) it",
    re.IGNORECASE,
)
_SEMI_AUTONOMOUS_SIGNAL = re.compile(
    r"show me|let me (see|approve|pick|choose|review)|before (you )?buy|"
    r"before (you )?purchas|approve (it|first|before|this)|check with me|"
    r"i'll (pick|choose|approve|decide)|"
    # Bugfix round 9 (issue 4): a live test showed "I want choose,review and
    # buy" correctly failed the grounding check (the LLM quoted its own
    # reply, not the user's words - see _mode_evidence_is_grounded's log
    # line) but ALSO fell through this fallback, since every pattern above
    # requires an explicit "let me"/"i'll" prefix. This phrase - "I want
    # (to) choose/review/pick/approve/decide", with no "you" between "want"
    # and the verb, so it's the USER doing the choosing/reviewing, not
    # delegating it to the agent - was missing from the enumeration.
    r"i want (to )?(see|approve|pick|choose|review|decide)",
    re.IGNORECASE,
)


def _mode_evidence_is_grounded(mode_evidence: Optional[str], raw_message: str) -> bool:
    """Bugfix round 8: the first line of defense against a hallucinated
    mode switch. The LLM is now required (see llm_client.py's prompt) to
    quote the exact words from the user's message that justify its
    desired_mode proposal. This is a cheap, deterministic check that the
    quote is actually real - a case-insensitive substring match against the
    raw message - not an attempt to judge whether the quote is a *good*
    justification (that's still the LLM's job). A missing/empty quote, or
    one that doesn't appear in the message at all (the model paraphrased,
    summarized, or invented it), fails this check."""
    if not mode_evidence:
        return False
    evidence = mode_evidence.strip().lower()
    if len(evidence) < 3:
        # Too short to be meaningful evidence (stray punctuation, a single
        # word fragment) - require at least a few characters of real quote.
        return False
    return evidence in raw_message.lower()


def _apply_desired_mode(
    session: dict,
    desired_mode: Optional[str],
    raw_message: str = "",
    mode_evidence: Optional[str] = None,
) -> Optional[str]:
    """V2 Batch 2 mode inference, reworked in bugfix round 8 for
    versatility and again in round 13 to fix a real independence gap
    round 8 introduced.

    Two signal sources are considered, and EITHER one alone can trigger a
    switch:
    1. The LLM's own proposal (`desired_mode`), trusted only when its
       `mode_evidence` quote is grounded - i.e. actually appears in the
       raw message (see `_mode_evidence_is_grounded`). This is what lets
       phrasing nobody enumerated in advance work (e.g. "go ahead and get
       it", "I'd rather look it over myself first") - round 8's whole
       point.
    2. A deterministic regex read of the RAW message
       (`_FULLY_AUTONOMOUS_SIGNAL` / `_SEMI_AUTONOMOUS_SIGNAL`), computed
       independently of whatever the LLM proposed this turn.

    Bugfix round 13: until now, signal 2 only ever ran as a FALLBACK,
    conditioned on the LLM having already proposed a mode switch that then
    failed grounding. If the LLM proposed nothing at all (`desired_mode is
    None`) or simply re-proposed the CURRENT mode - which happens more
    than expected once a mode is already established earlier in the
    conversation, the model seems to anchor on "we already decided this" -
    the regex never even ran, so a bare "buy it" said WHILE ALREADY in
    Semi-Autonomous mode could silently do nothing at all, even though
    that exact phrase has meant "switch to Fully Autonomous and just buy
    it" since round 1a. Signal 2 now runs unconditionally, every turn,
    regardless of what (if anything) the LLM proposed - a mixed message
    (both a buy word and a review/choose word) still prefers
    semi_autonomous, same tie-break the prompt already teaches the LLM.
    When both signals name a mode, the LLM's grounded proposal wins (it
    can be more specific than a blunt regex); the regex is what fires when
    the LLM's own classifier misses an otherwise-unambiguous phrase.

    This only ever flips a UI/session setting; it can never itself
    authorize a purchase (see UnderstandResult's docstring), so a wrong
    guess is a UX annoyance the human can correct with a mode button,
    never a spending risk.

    Bugfix round 15: the Recommendation mode this function used to be able
    to resolve to (exempt from grounding, since it only ever narrowed
    capability) is gone - Semi-Autonomous already does everything it did
    (search, reason, show candidates) plus buy-on-approval, so it was a
    strict subset with no purpose of its own. Only the two purchase-
    capable modes are resolvable now, both grounded the same way."""
    current_mode = session["autonomy_mode"]

    llm_mode: Optional[str] = None
    if desired_mode in ("semi_autonomous", "fully_autonomous") and _mode_evidence_is_grounded(
        mode_evidence, raw_message
    ):
        llm_mode = desired_mode

    fully_signal = bool(_FULLY_AUTONOMOUS_SIGNAL.search(raw_message))
    semi_signal = bool(_SEMI_AUTONOMOUS_SIGNAL.search(raw_message))
    if semi_signal:
        regex_mode: Optional[str] = "semi_autonomous"  # mixed signal -> prefer the more conservative mode
    elif fully_signal:
        regex_mode = "fully_autonomous"
    else:
        regex_mode = None

    resolved_mode = llm_mode or regex_mode
    if resolved_mode is None or resolved_mode == current_mode:
        if desired_mode not in (None, current_mode) and llm_mode is None and regex_mode is None:
            logger.warning(
                "Ignoring desired_mode=%s - mode_evidence %r not grounded in message and no "
                "regex-fallback signal matched either. Message: %r",
                desired_mode, mode_evidence, raw_message,
            )
        return None

    if llm_mode:
        logger.info(
            "Applying desired_mode=%s - grounded by LLM-quoted evidence %r in message: %r",
            resolved_mode, mode_evidence, raw_message,
        )
    else:
        logger.info(
            "Applying desired_mode=%s - regex signal matched independently of the LLM's own "
            "proposal (%r) in message: %r",
            resolved_mode, desired_mode, raw_message,
        )

    session["autonomy_mode"] = resolved_mode
    return resolved_mode


def _apply_product_switch_mode_reset(session: dict, category_switched: bool) -> Optional[str]:
    """Bugfix round 14 (an explicit design request, not a reported bug): a
    genuine product switch (see `_merge_context`'s `category_switched`,
    fired when the accumulated context is wiped because this turn's
    category clearly conflicts with the previous one) must not let a
    Fully-Autonomous stance - or an active delegated authority's spending
    power - silently carry over onto a brand new item the user never said
    anything autonomous about. Every genuine item switch resets the
    session back to the conservative Semi-Autonomous default.

    Called BEFORE `_apply_desired_mode` in the same turn, so the two
    compose naturally: this resets to the safe baseline first, then
    `_apply_desired_mode` can still promote it right back up to Fully
    Autonomous if THIS SAME message also carries its own fresh purchase-
    intent wording (e.g. "I also want a sports watch, just buy it") -
    exactly like every other guard in this module, nothing here overrides
    what the user explicitly says in the turn that switched items.

    Returns the new mode ("semi_autonomous") only when this call actually
    changed something (i.e. the session wasn't already semi_autonomous),
    so a turn that needed no reset doesn't report a no-op mode switch to
    the frontend."""
    if not category_switched:
        return None
    if session["autonomy_mode"] == "semi_autonomous":
        return None
    logger.info(
        "Product switch detected - resetting autonomy mode to semi_autonomous (was %s)",
        session["autonomy_mode"],
    )
    session["autonomy_mode"] = "semi_autonomous"
    return "semi_autonomous"


# Bugfix round 7: the local model has been seen to mishandle an explicit
# two-number budget range when the user states it high-then-low (e.g.
# "budget is between 5k to 2k") - it correctly extracts a range when the
# numbers are given low-then-high ("2k to 5k"), but the reversed phrasing
# gets misread as a single max_price (only the second number survives,
# the first is silently dropped), leaving a search too narrow to match
# anything. Per this codebase's established pattern of parsing money
# amounts deterministically from the raw text rather than trusting the
# LLM's own numeric extraction (see budget-floor-vs-ceiling handling
# elsewhere in this module), a range is now additionally recognized with a
# regex against the raw message - order-independent, always assigning the
# smaller number to min_price and the larger to max_price - and overrides
# whatever the LLM extracted for this turn whenever the message actually
# contains a two-number range pattern. A plain single-number budget
# ("under 5000") doesn't match this pattern at all, so is untouched and
# still relies on the LLM/prompt rules as before.
_MONEY_TOKEN = r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(k)?\b"
_BUDGET_RANGE_PATTERN = re.compile(
    r"(?:between\s+)?" + _MONEY_TOKEN + r"\s*(?:to|-|and|~)\s*" + _MONEY_TOKEN,
    re.IGNORECASE,
)


def _parse_money_token(number_str: str, k_suffix: Optional[str]) -> float:
    value = float(number_str.replace(",", ""))
    if k_suffix:
        value *= 1000
    return value


def _apply_budget_range_guard(intent: ShoppingContext, raw_message: str) -> ShoppingContext:
    """If the raw message contains an explicit two-number budget range
    (in either order), override this turn's min_price/max_price with the
    correctly sorted values - see the module comment above for why this
    isn't left to the LLM alone. Returns the same intent object (mutated)
    for convenience at the call site.

    Guarded against firing on an unrelated numeric range (a shoe size
    range, an age range, etc.) by requiring either a currency/money marker
    (a "k" suffix, "rs"/"inr"/"₹") on at least one number, or the word
    "budget"/"price"/"cost" somewhere in the message - a bare "size 8 to
    9" or "ages 5 to 8" won't match either, so it's left to the LLM/prompt
    rules as before."""
    match = _BUDGET_RANGE_PATTERN.search(raw_message)
    if not match:
        return intent
    has_money_marker = bool(match.group(2) or match.group(4)) or bool(
        re.search(r"rs\.?|inr|₹", match.group(0), re.IGNORECASE)
    )
    has_budget_word = bool(re.search(r"budget|price|cost", raw_message, re.IGNORECASE))
    if not (has_money_marker or has_budget_word):
        return intent
    a = _parse_money_token(match.group(1), match.group(2))
    b = _parse_money_token(match.group(3), match.group(4))
    lo, hi = min(a, b), max(a, b)
    if intent.min_price != lo or intent.max_price != hi:
        logger.info(
            "Budget range guard: overriding LLM-extracted budget (min=%s, max=%s) with "
            "(min=%s, max=%s) parsed directly from message: %r",
            intent.min_price, intent.max_price, lo, hi, raw_message,
        )
    intent.min_price = lo
    intent.max_price = hi
    return intent


# Bugfix round 9 (issue j, piece 2): the same class of bug as the range
# guard above, but for a plain single-number ceiling ("under 5000rs",
# "budget 4000", "max 3000") - live testing showed the local model
# sometimes returns max_price=null even when a number like this is stated
# plainly in the message (confirmed via backend logs: no guard fired
# either way for a message that should have produced max_price=5000, and
# the identical phrasing worked correctly later in the same session - a
# same-turn extraction slip, not a deterministic bug). Only fills
# max_price when the LLM's own extraction came back empty - never
# overrides a value the model DID extract, unlike the range guard above
# (which always wins once a genuine two-number range is found, since a
# range is much less ambiguous to detect than a single trigger word).
_SINGLE_BUDGET_PATTERN = re.compile(
    r"(?:under|within|up ?to|max(?:imum)?|below|less than|budget(?:\s+is|\s+of|\s+would be)?|"
    r"around|about)\s*(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(k)?\s*(?:rs\.?|inr|rupees)?",
    re.IGNORECASE,
)


def _apply_single_budget_guard(intent: ShoppingContext, raw_message: str) -> ShoppingContext:
    """If the LLM left max_price null but the raw message plainly states a
    single-number ceiling ("under 5000rs"), fill it in directly from the
    message - see the module comment above. Guarded the same way as the
    range guard: only fires when there's a currency/"k" marker on the
    number or the word budget/price/cost somewhere in the message, so a
    bare "under 9" (a shoe size) is left alone."""
    if intent.max_price is not None:
        return intent
    match = _SINGLE_BUDGET_PATTERN.search(raw_message)
    if not match:
        return intent
    has_money_marker = bool(match.group(2)) or bool(
        re.search(r"rs\.?|inr|₹|rupees", match.group(0), re.IGNORECASE)
    )
    has_budget_word = bool(re.search(r"budget|price|cost", raw_message, re.IGNORECASE))
    if not (has_money_marker or has_budget_word):
        return intent
    value = _parse_money_token(match.group(1), match.group(2))
    logger.info(
        "Single-budget guard: LLM left max_price null, filling in %s parsed directly "
        "from message: %r",
        value, raw_message,
    )
    intent.max_price = value
    return intent


# Bugfix round 9 (issue j, piece 3): the category counterpart to the
# budget guards above. The local model occasionally returns category=null
# even when the message (or that turn's own extracted keywords) literally
# contains one of the real catalog category strings verbatim - same
# same-turn extraction-slip failure mode, confirmed the same way (no
# "dropping unrecognized category" log line, meaning the model returned
# null itself rather than an invalid value getting dropped). Only fills
# category when the LLM's own extraction came back empty.
def _apply_category_fallback(intent: ShoppingContext, raw_message: str) -> ShoppingContext:
    if intent.category:
        return intent
    haystack = raw_message.lower()
    keyword_haystack = " ".join(intent.keywords).lower() if intent.keywords else ""
    for cat in KNOWN_CATEGORIES:
        cat_l = cat.lower()
        if cat_l in haystack or cat_l in keyword_haystack:
            logger.info(
                "Category fallback: LLM left category null, found %r verbatim in "
                "message/keywords: %r",
                cat, raw_message,
            )
            intent.category = cat
            return intent
    return intent


# Bugfix round 3 follow-up: catches a reply that is STILL asking for more
# information even though it doesn't end in a literal "?" - e.g. "Sorry, I
# need more information about the brand or style to give you accurate
# results." Seen live: this reply doesn't end in "?", so the plain
# ends-with-"?" check let it slip through the backstop below as if it were
# a confirmation, forcing a search that then genuinely succeeded (the
# accumulated context really was enough) - but the user briefly saw a
# "need more info" bubble immediately followed by real results, which
# reads as broken. This is a lightweight, English-specific heuristic (not
# perfect), used only to decide whether the reply text should be replaced
# with a clean transitional line before showing it - it never affects
# whether we actually search, which is still decided purely from context.
_STILL_ASKING_PATTERNS = re.compile(
    r"\?|more information|let me know|could you|can you|please (tell|specify|share|provide)|"
    r"what (is|are|type|kind|style|brand)|which (brand|style|type|one)|not sure|need to know|"
    r"sorry,? i need",
    re.IGNORECASE,
)


# Bugfix round 6: the user explicitly decided the agent should always
# gather a category AND a budget together before its first search,
# instead of searching as soon as it has a category alone (the previous
# behavior - see `_STILL_ASKING_PATTERNS`'s round-3 note for the earlier,
# looser version of this check). This is the one explicit override that
# still lets a search proceed without a budget: the user telling it not to
# bother asking.
_EXPLICIT_SEARCH_NOW = re.compile(
    r"search now|go ahead( and search)?|just (show|find|search)|proceed( now)?|"
    r"any budget( is fine)?|no budget|don'?t (worry about|need) (the )?budget",
    re.IGNORECASE,
)

# Bugfix round 9 (issue m): detects a reply that claims a search is
# happening right now ("Got it - searching for a sports watch now.") - see
# _apply_ready_to_search_backstop's docstring for why this matters on the
# "force a search OFF" path specifically.
_CLAIMS_SEARCHING_NOW = re.compile(
    r"\bsearch(?:ing)?\b[^.?!]{0,60}\bnow\b",
    re.IGNORECASE,
)


def _apply_ready_to_search_backstop(
    ready_to_search: bool,
    context: ShoppingContext,
    reply: str,
    raw_message: str = "",
) -> tuple[bool, Optional[str]]:
    """V2 Batch 2 bugfix, broadened after a second live-test round, with a
    third-round follow-up, and now a sixth-round tightening. Per this
    codebase's established pattern (see `_merge_context`'s stale-min_price
    guard, and the recommendation agent's hallucination guard), we don't
    fully trust the LLM's own `ready_to_search` JSON either way - this
    function is the single source of truth for whether a search actually
    fires, in BOTH directions:

    - Forces a search the LLM didn't flag as ready when the accumulated
      context is already enough (rounds 1-3): the local model sometimes
      gets its own prose reply right ("Got it - searching for X now")
      while failing to also flip `ready_to_search` to true, or its reply
      still reads like a clarifying question even though we've already
      decided (from verified context) to search anyway - see
      `_STILL_ASKING_PATTERNS`'s docstring for that half.
    - Round 6, new: also REFUSES a search the LLM DID flag as ready when
      the context isn't actually enough by the user's own bar - a category
      AND a budget (min_price or max_price), not just a category alone.
      Live testing showed the model searching the instant it had a
      category, skipping budget entirely even when the one-clarifying-
      round rule in the prompt hadn't been used up yet. The one explicit
      escape hatch is the user telling it to skip budget and search
      anyway (`_EXPLICIT_SEARCH_NOW`, e.g. "search now", "any budget is
      fine") - respecting an explicit instruction, same as before.

    This is safe to apply in both directions because the frontend has its
    own independent safety net against over-firing either way: `App.tsx`
    only actually schedules an auto-search when the accumulated context's
    signature has changed since the last auto-search, so a "ready" verdict
    that adds nothing new never re-triggers a search here.

    Returns `(ready_to_search, reply_override)`. `reply_override` is
    non-None only when we're forcing a search ON and the model's own reply
    text still reads like it's asking for more - in that case the
    contradictory "I need more info" text is replaced with a plain
    transitional line before it ever reaches the user, since we've already
    decided (from verified context, not the model's judgment) to search
    anyway; showing the model's un-listened-to question right before real
    results appear would just be confusing.

    Bugfix round 9 (issue m): the docstring used to say no override is
    needed when we force a search OFF, on the assumption that "the backend
    silently declining to search yet" never contradicts anything the user
    can see. Live testing proved that wrong: after a topic-conflict reset
    (e.g. shoes -> a smartwatch), the model's reply said "Got it -
    searching for a sports watch now" while this function correctly
    withheld the search (new category, no budget yet) - the backend's
    silent "not yet" was directly contradicted by the model's own claim,
    and the user reasonably concluded the feature was broken and reset the
    conversation. Now the OFF path also gets a reply_override, but only
    when the model's reply actually claims a search is happening
    (`_CLAIMS_SEARCHING_NOW`) - an honest question for whatever's actually
    missing (category, budget, or both) replaces that false claim, instead
    of leaving a promise the backend has no intention of keeping.
    """
    has_category_and_budget = bool(context.category) and (
        context.min_price is not None or context.max_price is not None
    )
    explicit_override = bool(_EXPLICIT_SEARCH_NOW.search(raw_message))
    ready_by_context = has_category_and_budget or (bool(context.category) and explicit_override)
    reply_still_asking = bool(_STILL_ASKING_PATTERNS.search(reply))

    if ready_by_context:
        if reply_still_asking:
            return True, "Got it — I already have enough to go on. Searching now…"
        return True, None

    # Not enough context by the user's own bar (category + budget), and no
    # explicit "search now" override - never search yet, even if the LLM's
    # own `ready_to_search` said true. Worst case, the user can still hit
    # the manual "Search now" button.
    if _CLAIMS_SEARCHING_NOW.search(reply):
        if not context.category:
            override = "Before I search, what are you looking for, and what's your budget?"
        else:
            override = f"Before I search for {context.category}, what's your budget?"
        logger.info(
            "Ready-to-search backstop: reply falsely claimed a search was happening "
            "(%r) while withholding it (missing category and/or budget) - replacing "
            "with an honest question instead.",
            reply,
        )
        return False, override
    return False, None


def _find_actionable_order(db, session_id: str) -> Optional[Order]:
    """Bugfix round 10: the most recent order for this session that is
    still awaiting an outcome - selected but not yet checked out
    (SELECTED), or checked out and Guardian-approved but still awaiting a
    manual payment click (PAYMENT_PENDING). PAID/FAILED orders are done
    and never block a new search. Used by
    `_apply_order_in_progress_guard` below."""
    return (
        db.query(Order)
        .filter(Order.session_id == session_id)
        .filter(Order.status.in_(["SELECTED", "PAYMENT_PENDING"]))
        .order_by(Order.id.desc())
        .first()
    )


def _apply_order_in_progress_guard(
    ready_to_search: bool,
    reply_override: Optional[str],
    context: ShoppingContext,
    db,
    session_id: str,
    autonomy_mode: str = "semi_autonomous",
) -> tuple[bool, Optional[str]]:
    """Bugfix round 10: "Buy it" and "I want you to buy it", sent while an
    order for the SAME item is already SELECTED or PAYMENT_PENDING, were
    both silently re-triggering a brand new product search instead of
    doing anything with the order sitting there. Root cause: nothing in
    the chat pipeline ever looked at order state at all - `ready_to_search`
    is decided purely from context completeness (see
    `_apply_ready_to_search_backstop`), and the LLM (correctly) sees the
    category/budget are already known from earlier in the conversation, so
    it comes back true every time, regardless of whether an order already
    exists for that exact context.

    This only ever SUPPRESSES a new search - it never touches the order,
    the Guardian, or the payment flow itself. Completing the order is
    still either the user's own "Pay now" click (Semi-Autonomous) or the
    existing fully_autonomous auto-flow in App.tsx (unchanged), consistent
    with the Payment Guardian being the only thing that can move money.

    If the accumulated context's category no longer matches the pending
    order's product (the user has moved on to a different item, e.g.
    shoes -> a smartwatch), this does nothing - that is the legitimate
    sequential "finish one, shop for the next" flow from bugfix round 9
    (issue m), which already works and must keep working.

    Bugfix round 12: the nudge text is now mode-aware. In
    Semi-Autonomous mode there IS something for the user to do (click
    "Pay now" / use the Order panel), so the original manual-action
    phrasing stays. In Fully-Autonomous mode there is nothing for the
    user to click - App.tsx's own auto-flow `useEffect` chain is already
    progressing this exact order on its own (checkout, then payment) -
    so telling the user to "use the Order panel" there was actively
    misleading (it reads like a Semi-Autonomous instruction). The
    fully-autonomous phrasing instead just confirms it's already in
    motion."""
    if not ready_to_search:
        return ready_to_search, reply_override
    order = _find_actionable_order(db, session_id)
    if order is None:
        return ready_to_search, reply_override

    product = db.get(Product, order.product_id)
    order_category = (product.category if product else "").strip().lower()
    current_category = (context.category or "").strip().lower()
    if order_category and current_category and order_category != current_category:
        return ready_to_search, reply_override

    fully_auto = autonomy_mode == "fully_autonomous"
    if order.status == "PAYMENT_PENDING":
        if fully_auto:
            nudge = (
                f"Order #{order.id} ({order.product_name}) already passed the "
                f"Payment Guardian and is completing payment automatically — no "
                f"action needed."
            )
        else:
            nudge = (
                f'You already have Order #{order.id} ({order.product_name}) waiting on '
                f'payment — click "Pay now (Razorpay)" below to complete it, or tell me '
                f"you'd like to shop for something else."
            )
    else:
        if fully_auto:
            nudge = (
                f"Order #{order.id} ({order.product_name}) is already selected — "
                f"the Payment Guardian is running automatically, no action needed."
            )
        else:
            nudge = (
                f"You already have Order #{order.id} ({order.product_name}) selected and "
                f"awaiting checkout — use the Order panel to continue, or tell me you'd "
                f"like to shop for something else."
            )
    logger.info(
        "Order-in-progress guard: suppressing a new search for session %s - "
        "order #%s is still %s for the same category (%r), mode=%s.",
        session_id, order.id, order.status, order_category or current_category, autonomy_mode,
    )
    return False, nudge


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "context": ShoppingContext(),
            # Bugfix round 15: a fresh session used to start in
            # Recommendation mode. That mode is gone (Semi-Autonomous
            # already does everything it did, plus buy-on-approval), so
            # Semi-Autonomous is now the conservative default a new
            # session starts in - the user still has to explicitly say or
            # click their way into Fully Autonomous.
            "autonomy_mode": "semi_autonomous",
        }
    return _sessions[session_id]


def _categories_conflict(old_category: str, new_category: str) -> bool:
    """True when two category strings clearly describe different product
    types rather than just different phrasing of the same one (e.g.
    'running shoe' vs 'running shoes' should NOT conflict)."""
    old_l, new_l = old_category.lower().strip(), new_category.lower().strip()
    if old_l == new_l:
        return False
    return old_l not in new_l and new_l not in old_l


def _merge_context(base: ShoppingContext, turn: ShoppingContext) -> tuple[ShoppingContext, bool]:
    """Fold this turn's extracted intent into the accumulated context.

    Scalars: a new non-null value overrides the old one (the user is
    refining/changing their mind). Lists/dicts: merge additively.

    Topic switch: if this turn states a category that clearly conflicts
    with the accumulated one (e.g. the user pivots from "sports watch" to
    "running shoes"), the old keywords/attributes/budget almost certainly
    describe the PREVIOUS product, not this one - carrying them forward
    would pollute the new search with irrelevant terms (this happened in
    practice: a watch's "rubbery strap, heart rate, GPS" keywords leaking
    into a shoe search). Start fresh from just this turn instead of merging.

    Returns `(merged_context, category_switched)` - the caller uses the
    bool to reset the session's autonomy mode back to semi_autonomous on a
    genuine product switch (see `_apply_product_switch_mode_reset` below),
    since a delegated/fully-autonomous authorization or "just buy it"
    stance from the PREVIOUS item should never silently carry over and
    auto-purchase a brand new one the user never actually said that about.
    """
    # Self-healing guard: the prompt requires "category" to be one of the
    # real catalog categories (or null), but a local model answering a
    # single-word clarifying question (e.g. the bot asked "neutral,
    # stability, or trail?" and the user just said "neutral") sometimes
    # echoes that word back as a brand-new "category" instead of leaving it
    # null or filing it as an attribute. Treating an invalid category as a
    # real topic switch would wipe the whole accumulated context over a
    # word that was never a category at all - so an unrecognized category
    # is dropped here before it can trigger a conflict-reset or override
    # anything, the same way llm_client.py already drops an unrecognized
    # desired_mode instead of trusting it.
    turn_category = turn.category
    if turn_category and turn_category.strip().lower() not in {c.lower() for c in KNOWN_CATEGORIES}:
        logger.warning("Dropping unrecognized category from LLM turn intent: %r", turn_category)
        turn_category = None

    category_switched = False
    if base.category and turn_category and _categories_conflict(base.category, turn_category):
        logger.info(
            "Category switch detected (%r -> %r) - resetting accumulated context",
            base.category,
            turn_category,
        )
        base = ShoppingContext()
        category_switched = True

    merged = base.model_copy(deep=True)

    if turn_category:
        merged.category = turn_category
    if turn.min_price is not None:
        merged.min_price = turn.min_price
    if turn.max_price is not None:
        merged.max_price = turn.max_price
    if turn.currency:
        merged.currency = turn.currency
    if turn.notes:
        merged.notes = turn.notes

    for kw in turn.keywords:
        if kw not in merged.keywords:
            merged.keywords.append(kw)

    merged.attributes.update(turn.attributes)

    # Self-healing guard: the LLM occasionally misreads a plain budget
    # statement as a floor (min_price) instead of a ceiling. If that ever
    # leaves min_price above max_price, the resulting range can never
    # match anything again for the rest of the session - drop the stale
    # min_price rather than silently returning empty results forever.
    if (
        merged.min_price is not None
        and merged.max_price is not None
        and merged.min_price > merged.max_price
    ):
        logger.warning(
            "Dropping stale min_price=%s (exceeds max_price=%s) after merge",
            merged.min_price,
            merged.max_price,
        )
        merged.min_price = None

    return merged, category_switched


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ollama_host": settings.ollama_host,
        "ollama_model": settings.ollama_model,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session = _get_session(req.session_id)
    history: list[ChatMessage] = session["history"]

    try:
        understood = await llm_client.understand(req.message, history, session["context"])
    except RuntimeError as exc:
        # Surface a clear, actionable error instead of a generic 500 - this
        # is almost always "Ollama isn't running" or "model not pulled".
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    _apply_budget_range_guard(understood.intent, req.message)
    _apply_single_budget_guard(understood.intent, req.message)
    _apply_category_fallback(understood.intent, req.message)
    session["context"], category_switched = _merge_context(session["context"], understood.intent)
    # Bugfix round 14: a genuine product switch resets the session back to
    # Semi-Autonomous BEFORE this turn's own mode-inference signal is
    # applied, so a leftover Fully-Autonomous stance (or active delegated
    # authority) from the PREVIOUS item never silently carries over onto a
    # new one - see `_apply_product_switch_mode_reset`'s docstring.
    product_switch_mode_reset = _apply_product_switch_mode_reset(session, category_switched)
    mode_changed = _apply_desired_mode(
        session, understood.desired_mode, req.message, understood.mode_evidence
    ) or product_switch_mode_reset
    ready_to_search, reply_override = _apply_ready_to_search_backstop(
        understood.ready_to_search, session["context"], understood.reply, req.message
    )
    db = get_db_session()
    try:
        ready_to_search, reply_override = _apply_order_in_progress_guard(
            ready_to_search, reply_override, session["context"], db, req.session_id,
            session["autonomy_mode"],
        )
    finally:
        db.close()
    reply = reply_override or understood.reply
    buy_confirmation = bool(_FULLY_AUTONOMOUS_SIGNAL.search(req.message))
    history.append(ChatMessage(role="user", content=req.message))
    history.append(ChatMessage(role="assistant", content=reply))

    return ChatResponse(
        reply=reply,
        turn_intent=understood.intent,
        context=session["context"],
        history=history,
        ready_to_search=ready_to_search,
        desired_mode=understood.desired_mode,
        mode_changed=mode_changed,
        buy_confirmation=buy_confirmation,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming counterpart to /api/chat - same logic, but each stage of
    understanding this turn (reading the message, extracting context,
    merging it into the accumulated context) is pushed live as its own
    SSE step, for the frontend's animated thought stream."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session = _get_session(req.session_id)
    history: list[ChatMessage] = session["history"]

    async def gen():
        yield _sse({
            "type": "step", "id": "read", "status": "done",
            "label": "Reading your message...",
        })
        await asyncio.sleep(STEP_PACE_SECONDS)
        yield _sse({
            "type": "step", "id": "understand", "status": "pending",
            "label": "Extracting shopping context from this message...",
        })
        try:
            understood = await llm_client.understand(req.message, history, session["context"])
        except RuntimeError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return

        turn_intent = understood.intent
        _apply_budget_range_guard(turn_intent, req.message)
        _apply_single_budget_guard(turn_intent, req.message)
        _apply_category_fallback(turn_intent, req.message)
        yield _sse({
            "type": "step", "id": "understand", "status": "done",
            "label": "Extracted this turn's context.",
            "detail": turn_intent.model_dump_json(indent=2),
        })
        await asyncio.sleep(STEP_PACE_SECONDS)

        merged, category_switched = _merge_context(session["context"], turn_intent)
        session["context"] = merged
        yield _sse({
            "type": "step", "id": "merge", "status": "done",
            "label": "Merged into accumulated shopping context.",
            "detail": merged.model_dump_json(indent=2),
        })
        await asyncio.sleep(STEP_PACE_SECONDS)

        # Bugfix round 14: a genuine product switch resets the session back
        # to Semi-Autonomous BEFORE this turn's own mode-inference signal is
        # applied - see `_apply_product_switch_mode_reset`'s docstring. The
        # two compose: this fires first, then `_apply_desired_mode` can
        # still promote it right back up if THIS message also carries its
        # own fresh purchase-intent wording.
        product_switch_mode_reset = _apply_product_switch_mode_reset(session, category_switched)
        # V2 Batch 2 - mode inference: only emitted as its own step when it
        # actually changes anything, so a turn with no autonomy signal
        # doesn't add noise to the thought stream.
        desired_mode_changed = _apply_desired_mode(
            session, understood.desired_mode, req.message, understood.mode_evidence
        )
        mode_changed = desired_mode_changed or product_switch_mode_reset
        if mode_changed:
            if desired_mode_changed:
                label = f"Detected autonomy intent — switching to {mode_changed.replace('_', ' ')} mode."
                detail = ("Inferred from this message's phrasing. Previous mode is still "
                          "available via the mode buttons if this wasn't what you meant.")
            else:
                label = "Product change detected — resetting to Semi-Autonomous mode for the new item."
                detail = ("A Fully-Autonomous stance (or an active delegated authority) from the "
                          "previous item never carries over automatically to a different one.")
            yield _sse({
                "type": "step", "id": "mode", "status": "done",
                "label": label,
                "detail": detail,
            })
            await asyncio.sleep(STEP_PACE_SECONDS)

        ready_to_search, reply_override = _apply_ready_to_search_backstop(
            understood.ready_to_search, merged, understood.reply, req.message
        )
        db = get_db_session()
        try:
            ready_to_search, reply_override = _apply_order_in_progress_guard(
                ready_to_search, reply_override, merged, db, req.session_id,
                session["autonomy_mode"],
            )
        finally:
            db.close()
        reply = reply_override or understood.reply
        buy_confirmation = bool(_FULLY_AUTONOMOUS_SIGNAL.search(req.message))
        if ready_to_search:
            yield _sse({
                "type": "step", "id": "ready", "status": "done",
                "label": "Enough context gathered — will search automatically.",
            })
            await asyncio.sleep(STEP_PACE_SECONDS)

        history.append(ChatMessage(role="user", content=req.message))
        history.append(ChatMessage(role="assistant", content=reply))

        yield _sse({
            "type": "result",
            "reply": reply,
            "turn_intent": json.loads(turn_intent.model_dump_json()),
            "context": json.loads(merged.model_dump_json()),
            "ready_to_search": ready_to_search,
            "desired_mode": understood.desired_mode,
            "mode_evidence": understood.mode_evidence,
            "mode_changed": mode_changed,
            "buy_confirmation": buy_confirmation,
        })

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset(req: ResetRequest):
    _sessions.pop(req.session_id, None)
    return {"status": "reset"}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = _get_session(session_id)
    return {"context": session["context"], "history": session["history"]}


@app.post("/api/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    session = _get_session(req.session_id)
    context: ShoppingContext = session["context"]

    if not context.category and not context.keywords:
        raise HTTPException(
            status_code=400,
            detail="Not enough context yet - tell the agent what you're shopping for first.",
        )

    try:
        result = await run_recommendation(llm_client, context)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RecommendResponse(
        reply=result.reply,
        candidates=[ProductOut(**c) for c in result.candidates],
        recommended_skus=result.recommended_skus,
        audit_log=[AuditEntryOut(step=a.step, detail=a.detail) for a in result.audit_log],
    )


@app.post("/api/recommend/stream")
async def recommend_stream(req: RecommendRequest):
    """Streaming counterpart to /api/recommend - the agent's own audit
    trail (Parsed request -> Searched products -> Retrieved candidates ->
    Compared products -> Selected recommendation) already IS the step
    sequence; this just pushes each entry the instant `run_recommendation_iter`
    produces it instead of only after the whole call finishes."""
    session = _get_session(req.session_id)
    context: ShoppingContext = session["context"]

    if not context.category and not context.keywords:
        raise HTTPException(
            status_code=400,
            detail="Not enough context yet - tell the agent what you're shopping for first.",
        )

    async def gen():
        step_index = 0
        try:
            async for item in run_recommendation_iter(llm_client, context):
                if isinstance(item, RecommendationResult):
                    yield _sse({
                        "type": "result",
                        "reply": item.reply,
                        "candidates": item.candidates,
                        "recommended_skus": item.recommended_skus,
                        "audit_log": [{"step": a.step, "detail": a.detail} for a in item.audit_log],
                    })
                else:
                    step_index += 1
                    # Round 7: a step with a stable `id` (the two
                    # "Reasoning" phases) is meant to be upserted in place
                    # by the frontend - pending placeholder, then replaced
                    # by the same id once real content exists - instead of
                    # appended as a new line each time, so its own id wins
                    # over the positional fallback used by every other step.
                    yield _sse({
                        "type": "step",
                        "id": item.id or f"recommend-{step_index}",
                        "status": item.status,
                        "label": item.step,
                        "detail": item.detail,
                    })
                    await asyncio.sleep(STEP_PACE_SECONDS)
        except RuntimeError as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/products", response_model=list[ProductOut])
async def list_products(category: str | None = None):
    """Debug/inspection endpoint - lets you sanity-check the seeded catalog
    directly (e.g. http://localhost:8000/api/products?category=running%20shoes)."""
    results = search_products(category=category, limit=100)
    return [ProductOut(**r) for r in results]

def _order_out(order: Order) -> OrderOut:
    return OrderOut(**order.to_dict())


def _authority_out(authority: DelegatedAuthority) -> AuthorityOut:
    return AuthorityOut(**authority.to_dict())


def _latest_authority(db, session_id: str) -> Optional[DelegatedAuthority]:
    return (
        db.query(DelegatedAuthority)
        .filter(DelegatedAuthority.session_id == session_id)
        .order_by(DelegatedAuthority.created_at.desc())
        .first()
    )


# V3: the names of the Guardian checks that mean "this fully-autonomous
# purchase was denied specifically for a delegated-authority problem" (no
# authority, revoked, expired, wrong category, insufficient budget) -
# every one of these is fixed by granting a fresh authority, so this is
# exactly the set of denials worth offering an in-chat "Grant authority"
# quick action for.
_AUTHORITY_CHECK_NAMES = {"Delegated authority", "Authority category", "Authority budget"}


def _authority_proposal(
    result: GuardianResult, order: Order, autonomy_mode: str, current_product: Optional[dict]
) -> Optional[dict]:
    """V3: when a fully-autonomous checkout is DENIED specifically because
    of the delegated-authority checks, propose concrete grant values for
    an in-chat "Grant authority" quick action - Claude's own
    permission-prompt pattern (the agent proposes, a human clicks to
    actually authorize). Deliberately backend-computed, never LLM-
    computed: the whole point of Delegated Authority is that spending
    limits come from a human, not the model, so even the *suggested*
    numbers here are derived only from trusted backend state (this
    order's own price/category), never anything the LLM said. The actual
    grant still only ever happens via a human clicking a real button that
    calls POST /api/authority/grant - this function only decides what
    values to pre-fill for them to review.

    Returns None when the denial has nothing to do with authority (e.g.
    price revalidation, out of stock) - granting a new authority wouldn't
    fix those, so there's nothing useful to propose.
    """
    if autonomy_mode != "fully_autonomous" or result.allowed:
        return None
    if not any(c.name in _AUTHORITY_CHECK_NAMES and not c.passed for c in result.checks):
        return None

    order_total = order.unit_price_at_selection * order.quantity
    return {
        # Sized exactly to this order - granting a fresh authority always
        # supersedes the old one for future Guardian checks (the Guardian
        # always reads the *latest* authority row), so a grant scoped to
        # just this purchase is enough to let it through, without handing
        # out a bigger standing budget than what was actually asked for.
        "suggested_max_spend": round(order_total, 2),
        "currency": order.currency,
        # Scoped to the specific product's category (principle of least
        # privilege) rather than "any" whenever we know it; falls back to
        # "any" only if the product lookup came back empty for some
        # reason - either way this is just a pre-filled suggestion the
        # human reviews before clicking Grant, never auto-applied.
        "category": (current_product or {}).get("category") or "any",
        "duration_minutes": 60,
        "reason": result.reason,
    }


def _credit_authority_spend(db, order: Order) -> None:
    """Called only on a CONFIRMED payment (real signature verified, or a
    stub simulated success) - never at Guardian ALLOW time. Reserving the
    budget at ALLOW would double-count an order that then fails payment;
    crediting it here means the delegated authority's remaining budget
    only ever reflects money that actually moved, which is the more
    honest number to show the user."""
    if not order.authority_id:
        return
    authority = db.get(DelegatedAuthority, order.authority_id)
    if authority is None:
        return
    order_total = order.unit_price_at_selection * order.quantity
    authority.spent += order_total
    order.log_event(
        "AUTHORITY_SPEND_RECORDED",
        f"Delegated authority #{authority.id} spend updated to {authority.currency} "
        f"{authority.spent:,.2f} of {authority.currency} {authority.max_spend:,.2f}.",
    )


@app.get("/api/categories")
async def list_categories():
    return {"categories": KNOWN_CATEGORIES}


@app.post("/api/authority/grant", response_model=AuthorityOut)
async def grant_authority(req: GrantAuthorityRequest):
    """Creates a new delegated-authority record for Fully-Autonomous mode.
    This is the ONLY way such a record comes into existence - the LLM has
    no path to create, raise, or extend one; only a human clicking "Grant
    authority" in the UI reaches this endpoint. Granting a new authority
    does not edit or delete an earlier one for this session - the Guardian
    (via `_latest_authority`) always looks at the most recent row, so a
    fresh grant is how an expired or revoked authority gets replaced,
    while the old record stays untouched in the audit trail."""
    if req.max_spend <= 0:
        raise HTTPException(status_code=400, detail="max_spend must be a positive amount.")
    if req.duration_minutes <= 0:
        raise HTTPException(status_code=400, detail="duration_minutes must be positive.")

    category = (req.category or "any").strip() or "any"
    if category.lower() != "any" and category.lower() not in [c.lower() for c in KNOWN_CATEGORIES]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category {category!r}. Use 'any' or one of: " + ", ".join(KNOWN_CATEGORIES),
        )

    db = get_db_session()
    try:
        authority = DelegatedAuthority(
            session_id=req.session_id,
            max_spend=req.max_spend,
            spent=0.0,
            currency="INR",
            category=category,
            revoked=False,
            expires_at=datetime.utcnow() + timedelta(minutes=req.duration_minutes),
        )
        db.add(authority)
        db.commit()
        db.refresh(authority)
        logger.info(
            "Session %s granted delegated authority #%s: max_spend=%.2f category=%s expires=%s",
            req.session_id, authority.id, authority.max_spend, authority.category, authority.expires_at,
        )
        return _authority_out(authority)
    finally:
        db.close()


@app.get("/api/authority")
async def get_authority(session_id: str):
    db = get_db_session()
    try:
        authority = _latest_authority(db, session_id)
        return _authority_out(authority).model_dump() if authority else None
    finally:
        db.close()


@app.post("/api/authority/revoke", response_model=AuthorityOut)
async def revoke_authority(req: RevokeAuthorityRequest):
    db = get_db_session()
    try:
        authority = _latest_authority(db, req.session_id)
        if authority is None:
            raise HTTPException(status_code=404, detail="No delegated authority found for this session.")
        authority.revoked = True
        db.commit()
        db.refresh(authority)
        logger.info("Session %s revoked delegated authority #%s.", req.session_id, authority.id)
        return _authority_out(authority)
    finally:
        db.close()


@app.get("/api/payment-config", response_model=PaymentConfigOut)
async def payment_config():
    gateway = get_payment_gateway()
    return PaymentConfigOut(
        mode=gateway.mode,
        key_id=settings.razorpay_key_id if gateway.mode == "real" else None,
    )


@app.post("/api/session/mode")
async def set_mode(req: SetModeRequest):
    # Bugfix round 15: Recommendation removed from the valid-mode list -
    # Semi-Autonomous is now the only non-fully-autonomous mode.
    if req.mode not in ("semi_autonomous", "fully_autonomous"):
        raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode!r}")
    session = _get_session(req.session_id)
    session["autonomy_mode"] = req.mode
    logger.info("Session %s autonomy mode set to %s", req.session_id, req.mode)
    return {"session_id": req.session_id, "autonomy_mode": req.mode}


@app.post("/api/orders/select", response_model=OrderOut)
async def select_order(req: SelectRequest):
    """SEARCHED -> SELECTED. Snapshots the product's current price/name/
    merchant onto the order - this is deliberately a point-in-time copy,
    not a live reference, so the Guardian has something fixed to
    revalidate the CURRENT product state against later."""
    session = _get_session(req.session_id)
    product = get_product_by_sku(req.sku)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Unknown sku: {req.sku}")
    if req.quantity < 1:
        raise HTTPException(status_code=400, detail="quantity must be at least 1")

    db = get_db_session()
    try:
        order = Order(
            session_id=req.session_id,
            product_id=product["id"],
            sku=product["sku"],
            product_name=product["name"],
            merchant=product["merchant"] or "",
            unit_price_at_selection=product["price"],
            currency=product["currency"],
            quantity=req.quantity,
            autonomy_mode=session["autonomy_mode"],
            status="SELECTED",
        )
        order.log_event("SELECTED", f"{product['name']} at {product['currency']} {product['price']:,.2f}")
        db.add(order)
        db.commit()
        db.refresh(order)
        return _order_out(order)
    finally:
        db.close()


@app.post("/api/orders/{order_id}/approve", response_model=OrderOut)
async def approve_order(order_id: int, req: ApproveRequest):
    """Explicit user approval for THIS order (semi-autonomous mode). This
    is separate from 'select' on purpose - selecting a product is not the
    same as authorizing payment for it, and the Guardian checks this flag
    independently rather than assuming selection implies consent."""
    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != req.session_id:
            raise HTTPException(status_code=404, detail="Order not found")
        order.user_approved = True
        order.log_event("USER_APPROVED", "User explicitly approved this order for payment.")
        db.commit()
        db.refresh(order)
        return _order_out(order)
    finally:
        db.close()


@app.post("/api/orders/{order_id}/checkout", response_model=CheckoutResponse)
async def checkout_order(order_id: int, req: CheckoutRequest):
    """Guardian gate, then (only on ALLOW) a real/stub Razorpay order is
    created. Nothing before this point has touched a payment gateway at
    all - this is the one and only place that can move from
    ALLOW/DENY into actually contacting Razorpay."""
    session = _get_session(req.session_id)
    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != req.session_id:
            raise HTTPException(status_code=404, detail="Order not found")

        current_product = get_product_by_sku(order.sku)
        authority = None
        if session["autonomy_mode"] == "fully_autonomous":
            authority = _latest_authority(db, req.session_id)
        result = guardian_evaluate(
            order=order,
            current_product=current_product,
            autonomy_mode=session["autonomy_mode"],
            user_approved=order.user_approved,
            authority=authority,
        )

        order.guardian_allowed = result.allowed
        order.guardian_reason = result.reason
        for check in result.checks:
            order.log_event(
                f"GUARDIAN_CHECK:{check.name}",
                f"{'PASS' if check.passed else 'FAIL'} - {check.detail}",
            )

        response = CheckoutResponse(
            order=_order_out(order),  # placeholder, replaced below after any status change
            guardian=GuardianResultOut(
                allowed=result.allowed,
                reason=result.reason,
                checks=[GuardianCheckOut(name=c.name, passed=c.passed, detail=c.detail) for c in result.checks],
            ),
        )

        if not result.allowed:
            order.status = "FAILED"
            order.log_event("GUARDIAN_DENIED", result.reason)
            db.commit()
            db.refresh(order)
            response.order = _order_out(order)
            proposal = _authority_proposal(result, order, session["autonomy_mode"], current_product)
            if proposal:
                response.authority_proposal = AuthorityProposalOut(**proposal)
            logger.info("Guardian DENIED order %s: %s", order_id, result.reason)
            return response

        # ALLOW - create the (real or stub) Razorpay order. Razorpay is
        # never called until this exact point.
        gateway = get_payment_gateway()
        amount_paise = int(round(order.unit_price_at_selection * order.quantity * 100))
        razorpay_order = gateway.create_order(
            amount_paise=amount_paise,
            currency=order.currency,
            receipt=f"agentbuyer_order_{order.id}",
        )
        order.razorpay_order_id = razorpay_order["id"]
        order.status = "PAYMENT_PENDING"
        if authority is not None:
            # Links this order to the specific authority record that
            # ALLOWed it, so a successful payment can later credit its
            # spend against THAT authority's budget (see verify_payment /
            # simulate_payment) rather than needing to re-look-up "the
            # latest one," which could have changed by then.
            order.authority_id = authority.id
        order.log_event(
            "ORDER_CREATED",
            f"Guardian ALLOWED - {gateway.mode} gateway order {razorpay_order['id']} created "
            f"for {order.currency} {amount_paise / 100:,.2f}.",
        )
        db.commit()
        db.refresh(order)

        response.order = _order_out(order)
        response.payment_mode = gateway.mode
        response.razorpay_order_id = razorpay_order["id"]
        response.amount_paise = amount_paise
        if gateway.mode == "real":
            response.razorpay_key_id = settings.razorpay_key_id
        return response
    finally:
        db.close()


@app.post("/api/orders/{order_id}/checkout/stream")
async def checkout_order_stream(order_id: int, req: CheckoutRequest):
    """Streaming counterpart to /api/orders/{id}/checkout - identical
    Guardian gate and identical post-ALLOW gateway order creation, but
    each individual GuardianCheck is pushed live as `evaluate_iter`
    decides it, instead of only after the whole evaluation finishes. This
    is the most important stream in the app: it's the live, visible proof
    that the backend - not the LLM - is deciding ALLOW/DENY, one named
    check at a time."""
    session = _get_session(req.session_id)
    db = get_db_session()

    async def gen():
        try:
            order = db.get(Order, order_id)
            if order is None or order.session_id != req.session_id:
                yield _sse({"type": "error", "message": "Order not found"})
                return

            current_product = get_product_by_sku(order.sku)
            authority = None
            if session["autonomy_mode"] == "fully_autonomous":
                authority = _latest_authority(db, req.session_id)

            result: Optional[GuardianResult] = None
            for item in guardian_evaluate_iter(
                order=order,
                current_product=current_product,
                autonomy_mode=session["autonomy_mode"],
                user_approved=order.user_approved,
                authority=authority,
            ):
                if isinstance(item, GuardianResult):
                    result = item
                else:
                    yield _sse({
                        "type": "step",
                        "id": f"guardian-{item.name}",
                        "status": "done" if item.passed else "error",
                        "label": f"Guardian: {item.name} — {'PASS' if item.passed else 'FAIL'}",
                        "detail": item.detail,
                    })
                    await asyncio.sleep(STEP_PACE_SECONDS)

            assert result is not None
            order.guardian_allowed = result.allowed
            order.guardian_reason = result.reason
            for check in result.checks:
                order.log_event(
                    f"GUARDIAN_CHECK:{check.name}",
                    f"{'PASS' if check.passed else 'FAIL'} - {check.detail}",
                )

            guardian_payload = {
                "allowed": result.allowed,
                "reason": result.reason,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.checks
                ],
            }

            if not result.allowed:
                order.status = "FAILED"
                order.log_event("GUARDIAN_DENIED", result.reason)
                db.commit()
                db.refresh(order)
                proposal = _authority_proposal(result, order, session["autonomy_mode"], current_product)
                yield _sse({
                    "type": "result",
                    "guardian": guardian_payload,
                    "order": json.loads(_order_out(order).model_dump_json()),
                    "payment_mode": None,
                    "razorpay_order_id": None,
                    "razorpay_key_id": None,
                    "amount_paise": None,
                    "authority_proposal": proposal,
                })
                logger.info("Guardian DENIED order %s: %s", order_id, result.reason)
                return

            # ALLOW - create the (real or stub) Razorpay order. Razorpay is
            # never called until this exact point.
            gateway = get_payment_gateway()
            amount_paise = int(round(order.unit_price_at_selection * order.quantity * 100))
            razorpay_order = gateway.create_order(
                amount_paise=amount_paise,
                currency=order.currency,
                receipt=f"agentbuyer_order_{order.id}",
            )
            order.razorpay_order_id = razorpay_order["id"]
            order.status = "PAYMENT_PENDING"
            if authority is not None:
                order.authority_id = authority.id
            order.log_event(
                "ORDER_CREATED",
                f"Guardian ALLOWED - {gateway.mode} gateway order {razorpay_order['id']} created "
                f"for {order.currency} {amount_paise / 100:,.2f}.",
            )
            db.commit()
            db.refresh(order)

            yield _sse({
                "type": "result",
                "guardian": guardian_payload,
                "order": json.loads(_order_out(order).model_dump_json()),
                "payment_mode": gateway.mode,
                "razorpay_order_id": razorpay_order["id"],
                "razorpay_key_id": settings.razorpay_key_id if gateway.mode == "real" else None,
                "amount_paise": amount_paise,
            })
        except Exception as exc:  # keep the stream alive long enough to report the failure
            logger.exception("checkout_stream failed for order %s", order_id)
            yield _sse({"type": "error", "message": str(exc)})
        finally:
            db.close()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/orders/{order_id}/simulate-payment", response_model=OrderOut)
async def simulate_payment(order_id: int, req: SimulatePaymentRequest):
    """Stub-gateway-only: stands in for the Razorpay Checkout callback so
    the full PAYMENT_PENDING -> PAID/FAILED transition can be demoed
    before real Test Mode keys are configured. Refuses to run once real
    keys are set, since a simulated outcome must never masquerade as a
    real payment result."""
    gateway = get_payment_gateway()
    if gateway.mode != "stub":
        raise HTTPException(
            status_code=400,
            detail="Real Razorpay keys are configured - use /verify-payment with a real "
            "payment id/signature instead of simulating one.",
        )
    if req.outcome not in ("success", "failure"):
        raise HTTPException(status_code=400, detail="outcome must be 'success' or 'failure'")

    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != req.session_id:
            raise HTTPException(status_code=404, detail="Order not found")
        if order.status != "PAYMENT_PENDING":
            raise HTTPException(
                status_code=400,
                detail=f"Order is in status {order.status}, not PAYMENT_PENDING - checkout it first.",
            )

        if req.outcome == "success":
            fake_payment_id = f"pay_stub_{uuid.uuid4().hex[:14]}"
            order.razorpay_payment_id = fake_payment_id
            order.status = "PAID"
            order.log_event("PAYMENT_SUCCESS", f"[stub] simulated successful payment {fake_payment_id}.")
            _credit_authority_spend(db, order)
        else:
            order.status = "FAILED"
            order.log_event("PAYMENT_FAILED", "[stub] simulated payment failure.")

        db.commit()
        db.refresh(order)
        return _order_out(order)
    finally:
        db.close()


@app.post("/api/orders/{order_id}/verify-payment", response_model=OrderOut)
async def verify_payment(order_id: int, req: VerifyPaymentRequest):
    """Real-gateway counterpart to /simulate-payment: called by the
    frontend after the Razorpay Checkout widget reports success. Verifies
    the payment signature with Razorpay's own utility before ever marking
    an order PAID - the frontend's word that "it succeeded" is not
    trusted, only a signature check against the key secret is."""
    gateway = get_payment_gateway()
    if gateway.mode != "real":
        raise HTTPException(
            status_code=400,
            detail="No real Razorpay keys are configured - use /simulate-payment instead.",
        )

    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != req.session_id:
            raise HTTPException(status_code=404, detail="Order not found")
        if order.status != "PAYMENT_PENDING":
            raise HTTPException(
                status_code=400,
                detail=f"Order is in status {order.status}, not PAYMENT_PENDING - checkout it first.",
            )
        if order.razorpay_order_id != req.razorpay_order_id:
            raise HTTPException(
                status_code=400,
                detail="razorpay_order_id does not match this order's checkout order.",
            )

        verified = gateway.verify_payment_signature(
            razorpay_order_id=req.razorpay_order_id,
            razorpay_payment_id=req.razorpay_payment_id,
            razorpay_signature=req.razorpay_signature,
        )

        if verified:
            order.razorpay_payment_id = req.razorpay_payment_id
            order.status = "PAID"
            order.log_event("PAYMENT_SUCCESS", f"Razorpay signature verified for payment {req.razorpay_payment_id}.")
            _credit_authority_spend(db, order)
            logger.info("Order %s PAID (Razorpay payment %s)", order_id, req.razorpay_payment_id)
        else:
            order.status = "FAILED"
            order.log_event("PAYMENT_FAILED", "Razorpay signature verification failed - payment not trusted.")
            logger.warning("Order %s signature verification FAILED", order_id)

        db.commit()
        db.refresh(order)
        return _order_out(order)
    finally:
        db.close()


@app.post("/api/orders/{order_id}/payment-attempt-failed", response_model=OrderOut)
async def payment_attempt_failed(order_id: int, req: PaymentAttemptFailedRequest):
    """Called by the frontend's Razorpay Checkout `payment.failed` handler
    (real gateway only) when a single card/UPI attempt is declined. This
    does NOT fail the order - Razorpay's own Checkout UI already offers
    the user a retry against the same order_id, and one declined card is
    not the same thing as the order having failed. It only records the
    attempt on the order's audit trail so the full history (including
    dead ends) stays visible, exactly like every other Guardian/order
    event in this system."""
    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != req.session_id:
            raise HTTPException(status_code=404, detail="Order not found")

        detail_parts = []
        if req.reason:
            detail_parts.append(f"reason={req.reason}")
        if req.description:
            detail_parts.append(req.description)
        if req.razorpay_payment_id:
            detail_parts.append(f"payment_id={req.razorpay_payment_id}")
        detail = "; ".join(detail_parts) or "Razorpay reported a failed payment attempt."

        order.log_event("PAYMENT_ATTEMPT_FAILED", detail)
        logger.info("Order %s: logged failed payment attempt (%s)", order_id, detail)

        db.commit()
        db.refresh(order)
        return _order_out(order)
    finally:
        db.close()


@app.get("/api/orders/{order_id}", response_model=OrderOut)
async def get_order(order_id: int, session_id: str):
    db = get_db_session()
    try:
        order = db.get(Order, order_id)
        if order is None or order.session_id != session_id:
            raise HTTPException(status_code=404, detail="Order not found")
        return _order_out(order)
    finally:
        db.close()


@app.get("/api/orders", response_model=list[OrderOut])
async def list_orders(session_id: str):
    db = get_db_session()
    try:
        orders = (
            db.query(Order)
            .filter(Order.session_id == session_id)
            .order_by(Order.created_at.desc())
            .all()
        )
        return [_order_out(o) for o in orders]
    finally:
        db.close()
