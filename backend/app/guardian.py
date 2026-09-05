"""
The Payment Guardian.

This is the single most important module in the whole project. Nothing
above this layer - not the LLM, not the recommendation agent, not the
user's chat messages - can authorize a payment by itself. Every check
here runs against data this module fetches or is handed directly by
trusted backend state (the Order row, a fresh DB read of the product) -
never against anything the LLM said about the order.

Checks, in order, matching the project spec's list:
  - autonomy mode allows purchasing at all
  - the product still exists
  - it's still in stock (quantity check)
  - currency is one this system supports
  - price hasn't silently increased since the user last saw/approved it
    (this is the "revalidate price/details" step from the order lifecycle
    diagram - it's what catches a bait-and-switch between selection and
    payment, and it applies in EVERY mode, not just Fully-Autonomous)
  - mode-specific authorization:
      (anything other than the two purchase-capable modes below)
                              -> always DENY, no purchase capability at all.
                              Bugfix round 15: this used to be a named
                              "recommendation" branch, back when
                              Recommendation was a real UI mode - it was
                              removed (Semi-Autonomous already subsumes
                              everything it did, plus the ability to buy
                              on approval, so it was a strict subset with
                              no purpose of its own). This check is kept
                              as an ALLOWLIST rather than deleted, so it
                              stays a defensive catch-all: the Guardian
                              denies any mode string it doesn't explicitly
                              recognize as purchase-capable, rather than
                              trusting upstream callers never to hand it
                              something unexpected (a stale mode value on
                              an old session, a future new mode nobody's
                              wired authorization for yet, etc).
      semi_autonomous      -> DENY unless the user has explicitly approved
                              THIS exact order (not just clicked "select")
      fully_autonomous      -> DENY unless an active, non-expired delegated
                              authority record exists for this session AND
                              this purchase's category and total are within
                              what it covers (max spend / category / expiry,
                              exactly as the spec calls for)

Razorpay is never called before this function returns ALLOW.

`evaluate_iter()` is the source of truth: a generator that yields each
GuardianCheck the INSTANT it's decided, so a streaming endpoint can show
the Guardian's reasoning live (V2 "thought stream" UI) instead of only
after the fact. The very last item it yields is always the final
GuardianResult (never a GuardianCheck), so a consumer can tell the
sequence is over. `evaluate()` below is the same generator fully drained
into one return value, kept for callers that don't need to stream.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

from .models import DelegatedAuthority, Order

SUPPORTED_CURRENCIES = {"INR"}


@dataclass
class GuardianCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GuardianResult:
    allowed: bool
    reason: str
    checks: list[GuardianCheck] = field(default_factory=list)


def evaluate_iter(
    order: Order,
    current_product: Optional[dict],
    autonomy_mode: str,
    user_approved: bool,
    authority: Optional[DelegatedAuthority] = None,
) -> Iterator[Union[GuardianCheck, GuardianResult]]:
    checks: list[GuardianCheck] = []

    def step(name: str, passed: bool, detail: str) -> GuardianCheck:
        c = GuardianCheck(name, passed, detail)
        checks.append(c)
        return c

    # 1. Autonomy mode gate - checked first because no other check matters
    # if this mode can never purchase at all. Bugfix round 15: an
    # allowlist of the two purchase-capable modes, not a check for the
    # single named mode ("recommendation") this used to deny - see the
    # module docstring's "Bugfix round 15" note above for why.
    if autonomy_mode not in ("semi_autonomous", "fully_autonomous"):
        yield step(
            "Autonomy mode", False,
            f"Mode {autonomy_mode!r} has no purchasing capability - DENY unconditionally.",
        )
        yield GuardianResult(False, f"{autonomy_mode!r} mode does not permit purchases.", checks)
        return
    yield step(
        "Autonomy mode", True, f"Mode '{autonomy_mode}' permits purchasing (subject to checks below)."
    )

    # 2. Product still exists in the catalog.
    if current_product is None:
        yield step("Product exists", False, f"sku {order.sku} no longer found in catalog.")
        yield GuardianResult(False, f"Product {order.sku} is no longer available.", checks)
        return
    yield step("Product exists", True, f"{order.sku} found in catalog.")

    # 3. Inventory.
    if current_product["stock_qty"] < order.quantity:
        yield step(
            "Inventory", False,
            f"Requested qty {order.quantity}, only {current_product['stock_qty']} in stock.",
        )
        yield GuardianResult(False, "Not enough stock to fulfill this order.", checks)
        return
    yield step(
        "Inventory", True, f"{current_product['stock_qty']} in stock, order needs {order.quantity}."
    )

    # 4. Currency sanity check.
    if current_product["currency"] not in SUPPORTED_CURRENCIES:
        yield step("Currency", False, f"Unsupported currency {current_product['currency']}.")
        yield GuardianResult(False, f"Unsupported currency: {current_product['currency']}.", checks)
        return
    yield step("Currency", True, f"{current_product['currency']} is supported.")

    # 5. Price revalidation - the product's CURRENT price must match what
    # the user actually approved. A merchant-side price change between
    # selection and payment is exactly the scenario the spec calls out
    # ("demonstrate price changes between search and purchase") - this is
    # a hard DENY, not a soft warning, regardless of autonomy mode.
    current_price = current_product["price"]
    if current_price > order.unit_price_at_selection:
        yield step(
            "Price revalidation", False,
            f"Price increased from {order.currency} {order.unit_price_at_selection:,.2f} "
            f"to {order.currency} {current_price:,.2f} since selection - approval is stale.",
        )
        yield GuardianResult(
            False,
            f"Price changed from {order.currency} {order.unit_price_at_selection:,.2f} to "
            f"{order.currency} {current_price:,.2f} since you selected this item - please "
            "re-select to review and approve the new price.",
            checks,
        )
        return
    yield step(
        "Price revalidation", True,
        f"Current price {order.currency} {current_price:,.2f} matches or is below the "
        f"{order.currency} {order.unit_price_at_selection:,.2f} that was approved.",
    )

    # 6. Mode-specific authorization.
    if autonomy_mode == "semi_autonomous":
        if not user_approved:
            yield step(
                "User approval", False, "Semi-autonomous mode requires explicit approval of this order."
            )
            yield GuardianResult(False, "Waiting for your explicit approval before this can be paid.", checks)
            return
        yield step("User approval", True, "User explicitly approved this order.")

    elif autonomy_mode == "fully_autonomous":
        # Delegated authority (max spend / category / expiry) is the ONLY
        # thing that can authorize a fully-autonomous purchase. Every
        # sub-check here is deliberately its own line in the audit trail,
        # matching the spec's own list (max spend, category, expiry) - a
        # DENY should always say exactly which of the three failed.
        if authority is None:
            yield step(
                "Delegated authority", False,
                "No delegated authority has ever been granted for this session.",
            )
            yield GuardianResult(
                False,
                "Fully-autonomous purchasing requires delegated authority (max spend / "
                "category / expiry) to be granted first - none has been set up yet.",
                checks,
            )
            return

        if authority.revoked:
            yield step("Delegated authority", False, f"Authority #{authority.id} was revoked.")
            yield GuardianResult(False, "The delegated authority for this session has been revoked.", checks)
            return

        now = datetime.datetime.utcnow()
        if authority.expires_at <= now:
            yield step(
                "Delegated authority", False,
                f"Authority #{authority.id} expired at {authority.expires_at.isoformat()}Z.",
            )
            yield GuardianResult(
                False,
                f"The delegated authority expired at {authority.expires_at.isoformat()}Z - "
                "grant a new one to continue autonomous purchasing.",
                checks,
            )
            return
        yield step(
            "Delegated authority", True,
            f"Authority #{authority.id} is active until {authority.expires_at.isoformat()}Z.",
        )

        allowed_category = authority.category.strip().lower()
        product_category = (current_product.get("category") or "").strip().lower()
        if allowed_category != "any" and allowed_category != product_category:
            yield step(
                "Authority category", False,
                f"Authority covers '{authority.category}', this purchase is "
                f"'{current_product.get('category')}'.",
            )
            yield GuardianResult(
                False,
                f"This purchase is in category '{current_product.get('category')}', which is "
                f"outside the delegated authority (covers only '{authority.category}').",
                checks,
            )
            return
        yield step(
            "Authority category", True,
            "Authority covers any category." if allowed_category == "any"
            else f"Authority covers '{authority.category}', matches this purchase.",
        )

        order_total = order.unit_price_at_selection * order.quantity
        remaining = authority.max_spend - authority.spent
        if order_total > remaining:
            yield step(
                "Authority budget", False,
                f"Order total {order.currency} {order_total:,.2f} exceeds remaining "
                f"{authority.currency} {remaining:,.2f} (of {authority.currency} "
                f"{authority.max_spend:,.2f} total, {authority.currency} {authority.spent:,.2f} "
                "already spent).",
            )
            yield GuardianResult(
                False,
                f"This purchase ({order.currency} {order_total:,.2f}) would exceed the "
                f"delegated authority's remaining budget of {authority.currency} "
                f"{remaining:,.2f} (of {authority.currency} {authority.max_spend:,.2f} total).",
                checks,
            )
            return
        yield step(
            "Authority budget", True,
            f"Order total {order.currency} {order_total:,.2f} is within remaining "
            f"{authority.currency} {remaining:,.2f} of {authority.currency} "
            f"{authority.max_spend:,.2f}.",
        )

    else:
        yield step("Autonomy mode", False, f"Unknown autonomy mode: {autonomy_mode!r}")
        yield GuardianResult(False, f"Unknown autonomy mode: {autonomy_mode!r}", checks)
        return

    yield GuardianResult(True, "All checks passed.", checks)


def evaluate(
    order: Order,
    current_product: Optional[dict],
    autonomy_mode: str,
    user_approved: bool,
    authority: Optional[DelegatedAuthority] = None,
) -> GuardianResult:
    """Non-streaming convenience wrapper: drains `evaluate_iter` and
    returns just the final GuardianResult, for callers that don't need
    the live per-check sequence."""
    result: Optional[GuardianResult] = None
    for item in evaluate_iter(order, current_product, autonomy_mode, user_approved, authority):
        if isinstance(item, GuardianResult):
            result = item
    assert result is not None
    return result
