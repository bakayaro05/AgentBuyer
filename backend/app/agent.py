"""
Recommendation agent (Recommendation autonomy mode only - no purchasing).

This is where "Discovery" and "Reasoning" actually happen, per the
project's architecture separation:

    Discovery  -> search_products() tool call against the real catalog
    Reasoning  -> LLM compares the *returned* candidates and picks favorites

The agent never has DB access itself - it can only see products via the
search_products tool, and it is instructed never to mention a product
that wasn't in the tool's results. If the model doesn't request the tool
on its own (small local models sometimes skip tool calls), the backend
forces a search using the accumulated context so "Searched products"
always genuinely happens rather than being skipped silently.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Union

from .llm_client import LLMClient
from .merchant_tools import SEARCH_PRODUCTS_TOOL, search_products
from .schemas import ShoppingContext

logger = logging.getLogger("agent_buyer.agent")

SYSTEM_PROMPT = """You are the product recommendation agent for Agent Buyer, a shopping \
assistant. You are in RECOMMENDATION-ONLY mode: you can search and suggest products, but \
you cannot purchase anything - there is no buying capability available to you at all.

You have one tool, search_products, which searches a real merchant catalog. Ground rules:
- Always call search_products before recommending anything, using the user's stated \
category, keywords, and budget.
- NEVER mention, describe, or recommend a product that was not returned by search_products. \
If you're unsure a product exists, don't mention it.
- Prices and stock in the tool results are real and current - never restate them differently.
- If nothing in the results fits well (e.g. everything is over budget), say so plainly and \
suggest what constraint to relax, instead of inventing a fitting product.

After you have search results, pick the best 1-3 matches for the user's stated budget and \
use case.

Give your final answer as STRICT JSON only, no markdown fences, matching this shape:
{
  "picks": [
    {"sku": "<sku - MUST be copied exactly from the search_products results above, \
never invented>", "why": "<one short sentence on why this fits, no product name or \
price needed here - the backend adds those from real data>"}
  ],
  "no_fit_reason": "<string explaining why nothing fits well, ONLY if you are including \
no picks - otherwise null>"
}

Critical: "sku" values are validated against the actual search results and anything that \
doesn't match is discarded before the user ever sees it - so there is no benefit to guessing \
or including a sku you are not certain came from the results. If you're unsure a product was \
in the results, leave it out of "picks" entirely.
"""


@dataclass
class AuditEntry:
    step: str
    detail: str
    # Round 7: `id`/`status` let a "Reasoning" step be yielded twice under
    # the SAME id - once as a pending placeholder before the (slow) LLM
    # call, once with the model's actual output once it resolves - so the
    # streaming thought-stream can update that one line in place instead
    # of only ever showing the generic placeholder. `id=None` (the
    # default) means "no stable identity, always append as its own line",
    # which is what every other step in this module still does.
    id: Optional[str] = None
    status: str = "done"


@dataclass
class RecommendationResult:
    reply: str
    candidates: list[dict]
    recommended_skus: list[str]
    audit_log: list[AuditEntry] = field(default_factory=list)


def _context_summary(context: ShoppingContext) -> str:
    parts = [f"category={context.category or 'unspecified'}"]
    if context.keywords:
        parts.append(f"keywords={', '.join(context.keywords)}")
    if context.min_price is not None or context.max_price is not None:
        parts.append(
            f"budget={context.min_price if context.min_price is not None else 'any'}"
            f"-{context.max_price if context.max_price is not None else 'any'} {context.currency}"
        )
    if context.attributes:
        parts.append(
            "attributes=" + ", ".join(f"{k}: {v}" for k, v in context.attributes.items())
        )
    if context.notes:
        parts.append(f"notes={context.notes}")
    return "; ".join(parts)


async def run_recommendation(
    llm: LLMClient, context: ShoppingContext
) -> RecommendationResult:
    """Non-streaming convenience wrapper: drains `run_recommendation_iter`
    and returns just the final RecommendationResult, for callers that
    don't need the live per-step sequence."""
    result: RecommendationResult | None = None
    async for item in run_recommendation_iter(llm, context):
        if isinstance(item, RecommendationResult):
            result = item
    assert result is not None
    return result


async def run_recommendation_iter(
    llm: LLMClient, context: ShoppingContext
) -> AsyncIterator[Union[AuditEntry, "RecommendationResult"]]:
    """Generator version: yields each AuditEntry the INSTANT it happens
    (Discovery then Reasoning, per the module's architecture split), so a
    streaming endpoint can show the agent's progress live (V2 "thought
    stream" UI). The very last item yielded is always the final
    RecommendationResult (never an AuditEntry)."""
    audit: list[AuditEntry] = []

    def step(
        step_name: str,
        detail: str,
        *,
        step_id: Optional[str] = None,
        status: str = "done",
        record: bool = True,
    ) -> AuditEntry:
        e = AuditEntry(step_name, detail, id=step_id, status=status)
        # A "pending" placeholder (e.g. the generic pre-LLM-call line) is
        # only for the live stream's benefit - it gets replaced in place by
        # a "done" entry with the same id once real content exists, so it
        # shouldn't also live on permanently in the returned audit_log
        # (that would show the generic text as its own separate, stale
        # line). `record=False` opts a yield out of the permanent list
        # without affecting what's streamed live.
        if record:
            audit.append(e)
        return e

    yield step("Parsed request", _context_summary(context))

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"User's shopping context so far: {_context_summary(context)}. "
            "Search for matching products and recommend the best options.",
        },
    ]

    yield step(
        "Reasoning",
        "Asking the model which products to search for...",
        step_id="reasoning-tools",
        status="pending",
        record=False,
    )

    first = await llm.raw_chat(messages, tools=[SEARCH_PRODUCTS_TOOL])
    tool_calls = first.get("tool_calls") or []

    # Round 7: show the model's ACTUAL output for this step instead of
    # leaving the generic placeholder as the only thing ever shown -
    # either its own reasoning text (if it returned any alongside the
    # tool call) or, more commonly for a tool-calling model, a summary of
    # the tool call it decided to make, which is itself a legitimate
    # window into "what the model inferred it should search for".
    first_reasoning_text = (first.get("content") or "").strip()
    if not first_reasoning_text:
        if tool_calls:
            fn0 = (tool_calls[0] or {}).get("function", {})
            first_reasoning_text = (
                f"Decided to call search_products with arguments: "
                f"{fn0.get('arguments')!r}"
            )
        else:
            first_reasoning_text = (
                "Model returned no reasoning text and did not call the search tool "
                "for this turn."
            )
    yield step("Reasoning", first_reasoning_text, step_id="reasoning-tools")

    candidates: list[dict] = []

    if tool_calls:
        messages.append(first)
        for call in tool_calls:
            fn = call.get("function", {})
            if fn.get("name") != "search_products":
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):  # some models stringify args
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            # Budget is a verified-context field, never the model's own
            # judgment: the model has shown it will invent plausible-looking
            # price bounds in its tool call even when the user never stated
            # any (seen in testing - it fabricated a 2000-8000 range out of
            # nowhere). category/keywords are low-stakes search refinements
            # and are fine to take from the model; min_price/max_price are
            # always taken from context, ignoring whatever the model put in
            # its own tool call arguments.
            ignored_price_args = {
                k: args[k] for k in ("min_price", "max_price") if k in args
            }
            search_args = {
                "category": args.get("category") or context.category,
                "keywords": args.get("keywords") or context.keywords or None,
                "min_price": context.min_price,
                "max_price": context.max_price,
            }
            results = search_products(**search_args)
            candidates.extend(results)
            detail = f"search_products({search_args}) -> {len(results)} candidate(s)"
            if ignored_price_args:
                detail += (
                    f" [model proposed price bounds {ignored_price_args} in its tool "
                    "call - ignored; budget is only ever taken from the verified context]"
                )
            yield step("Searched products", detail)
            messages.append({"role": "tool", "content": json.dumps(results)})
    else:
        # Model skipped the tool call - force a search from the accumulated
        # context so the audit trail (and the recommendation) is still
        # grounded in real data, per the "discovery must precede reasoning"
        # architecture rule.
        results = search_products(
            category=context.category,
            keywords=context.keywords or None,
            min_price=context.min_price,
            max_price=context.max_price,
        )
        candidates.extend(results)
        yield step(
            "Searched products",
            f"(agent did not call the tool; backend searched using accumulated "
            f"context directly) -> {len(results)} candidate(s)",
        )
        messages.append(
            {
                "role": "user",
                "content": f"search_products results: {json.dumps(results)}",
            }
        )

    yield step("Retrieved candidates", f"{len(candidates)} product(s) available to reason over")

    if not candidates:
        yield step("Compared products", "no candidates - nothing to compare")
        yield RecommendationResult(
            reply="I couldn't find anything matching that in the catalog right now — "
            "try widening the budget or category.",
            candidates=[],
            recommended_skus=[],
            audit_log=audit,
        )
        return

    messages.append(
        {
            "role": "user",
            "content": "Now give your final answer as the strict JSON shape described "
            "in your instructions, using only the products from the search results above.",
        }
    )

    yield step(
        "Reasoning",
        f"Comparing {len(candidates)} candidate(s) against your budget and use case...",
        step_id="reasoning-picks",
        status="pending",
        record=False,
    )

    final = await llm.raw_chat(messages, json_format=True)
    raw_content = final.get("content", "")
    # Round 7: same idea as the first reasoning step - show the model's
    # actual final-answer output (its raw JSON picks/why) rather than the
    # generic "Comparing N candidates..." placeholder, so clicking to
    # expand this step shows real content, not a canned line.
    yield step(
        "Reasoning",
        raw_content.strip() or "Model returned no content for its final answer this turn.",
        step_id="reasoning-picks",
    )

    candidates_by_sku = {c["sku"]: c for c in candidates}
    raw_picks: list[dict] = []
    no_fit_reason: str | None = None
    try:
        parsed = json.loads(raw_content)
        raw_picks = parsed.get("picks") or []
        no_fit_reason = parsed.get("no_fit_reason") or None
    except json.JSONDecodeError:
        logger.warning("Recommendation final answer was not valid JSON: %r", raw_content)

    # Ground truth check: the model's "picks" are NEVER trusted at face value.
    # Any sku it names that isn't actually in this turn's search results is a
    # hallucination (observed in testing - the model once recommended a
    # "PulseFit Pro" that was never returned by search_products) and gets
    # silently dropped here rather than reaching the user as fabricated text.
    valid_picks: list[dict] = []
    hallucinated_skus: list[str] = []
    for pick in raw_picks:
        sku = pick.get("sku") if isinstance(pick, dict) else None
        if sku and sku in candidates_by_sku:
            valid_picks.append({"sku": sku, "why": (pick.get("why") or "").strip()})
        elif sku:
            hallucinated_skus.append(sku)

    if hallucinated_skus:
        yield step(
            "Discarded hallucinated pick(s)",
            f"model referenced sku(s) not in the search results and they were "
            f"removed before reaching you: {', '.join(hallucinated_skus)}",
        )
        logger.warning(
            "Model recommended sku(s) not present in search results: %s", hallucinated_skus
        )

    recommended_skus = [p["sku"] for p in valid_picks]

    # The backend - not the model - writes the actual reply text, using only
    # verified product data (name/brand/price straight from the DB row) for
    # anything that names or prices a product. The model's "why" is still
    # used, but only as an explanatory clause, never as the source of a
    # product's identity.
    if valid_picks:
        lines = []
        for pick in valid_picks:
            product = candidates_by_sku[pick["sku"]]
            line = f"{product['name']} ({product['currency']} {product['price']:,.0f})"
            if pick["why"]:
                line += f" — {pick['why']}"
            lines.append(line)
        reply = "Here's what I found: " + "; ".join(lines) + "."
    else:
        reply = no_fit_reason or (
            "I couldn't confidently match anything in the results to your "
            "request — try widening the budget or category."
        )

    yield step("Compared products", f"model reasoned over {len(candidates)} candidate(s)")
    yield step(
        "Selected recommendation",
        f"{len(recommended_skus)} product(s) highlighted: {', '.join(recommended_skus) or 'none'}",
    )

    yield RecommendationResult(
        reply=reply,
        candidates=candidates,
        recommended_skus=recommended_skus,
        audit_log=audit,
    )
