"""
Pydantic models shared across the API.

`ShoppingContext` is the structured representation of "what the user wants"
that the LLM is asked to extract/update on every turn. This is the piece
we want to be able to *see* in the frontend to verify the LLM is actually
understanding the request correctly, before we ever let it search or
recommend products.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ShoppingContext(BaseModel):
    category: Optional[str] = Field(
        default=None, description="Product category, e.g. 'running shoes'"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Free-form search keywords/phrases"
    )
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: str = "INR"
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Key/value product attributes the user mentioned, "
        "e.g. {'use_case': 'marathon training', 'size': '9'}",
    )
    notes: Optional[str] = Field(
        default=None, description="Anything else worth remembering that doesn't fit above"
    )


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    turn_intent: ShoppingContext  # what was extracted from THIS message alone
    context: ShoppingContext  # accumulated/merged context across the conversation
    history: list[ChatMessage]
    # V2 Batch 2: the LLM's own proposal, this turn, for whether there's now
    # enough to search and which autonomy mode the phrasing implies. Neither
    # field has any spending power by itself - see UnderstandResult's
    # docstring in llm_client.py.
    ready_to_search: bool = False
    desired_mode: Optional[str] = None
    # Set only when the backend actually acted on desired_mode this turn
    # (i.e. it named a valid mode different from the session's current
    # one) - lets the frontend announce a real switch without re-deriving
    # the "did it actually change" logic itself.
    mode_changed: Optional[str] = None
    # Bugfix round 11: a deterministic (regex-based, same signal used for
    # fully_autonomous mode inference) read of whether THIS message itself
    # is a bare purchase confirmation ("buy it", "I want you to buy it").
    # The chat endpoint has no memory of candidates/orders (that state is
    # frontend-only), so it can't decide by itself whether this should
    # select a product - it just reports the raw signal and lets the
    # frontend decide what "buy it" should do given what's actually on
    # screen (an already-recommended candidate with no order yet, vs. an
    # order already in flight, which the order-in-progress guard handles
    # separately).
    buy_confirmation: bool = False


class ResetRequest(BaseModel):
    session_id: str


class ProductOut(BaseModel):
    id: int
    sku: str
    merchant: Optional[str] = None
    name: str
    brand: str
    category: str
    description: str
    price: float
    currency: str
    rating: float
    review_count: int
    in_stock: bool
    stock_qty: int
    attributes: dict[str, str]


class AuditEntryOut(BaseModel):
    step: str
    detail: str


class RecommendRequest(BaseModel):
    session_id: str


class RecommendResponse(BaseModel):
    reply: str
    candidates: list[ProductOut]
    recommended_skus: list[str]
    audit_log: list[AuditEntryOut]


class SetModeRequest(BaseModel):
    session_id: str
    mode: str  # "semi_autonomous" | "fully_autonomous" (Bugfix round 15: Recommendation removed)


class SelectRequest(BaseModel):
    session_id: str
    sku: str
    quantity: int = 1


class ApproveRequest(BaseModel):
    session_id: str


class CheckoutRequest(BaseModel):
    session_id: str


class SimulatePaymentRequest(BaseModel):
    session_id: str
    outcome: str  # "success" | "failure"


class VerifyPaymentRequest(BaseModel):
    session_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class PaymentAttemptFailedRequest(BaseModel):
    session_id: str
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    reason: Optional[str] = None
    description: Optional[str] = None


class GrantAuthorityRequest(BaseModel):
    session_id: str
    max_spend: float
    category: str = "any"
    duration_minutes: int = 60


class RevokeAuthorityRequest(BaseModel):
    session_id: str


class AuthorityOut(BaseModel):
    id: int
    max_spend: float
    spent: float
    remaining: float
    currency: str
    category: str
    revoked: bool
    expired: bool
    active: bool
    created_at: str
    expires_at: str


class GuardianCheckOut(BaseModel):
    name: str
    passed: bool
    detail: str


class GuardianResultOut(BaseModel):
    allowed: bool
    reason: str
    checks: list[GuardianCheckOut]


class OrderEventOut(BaseModel):
    step: str
    detail: str
    at: str


class OrderOut(BaseModel):
    id: int
    sku: str
    product_name: str
    merchant: str
    unit_price_at_selection: float
    currency: str
    quantity: int
    autonomy_mode: str
    status: str
    user_approved: bool
    guardian_allowed: Optional[bool] = None
    guardian_reason: Optional[str] = None
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    authority_id: Optional[int] = None
    events: list[OrderEventOut]
    created_at: str
    updated_at: str


class AuthorityProposalOut(BaseModel):
    """V3: when a fully-autonomous checkout is denied specifically for a
    missing/expired/revoked/insufficient delegated authority, the backend
    proposes concrete grant values (never the LLM - see main.py's
    `_authority_proposal()`) so the frontend can offer an in-chat
    "Grant authority" quick action, Claude-permission-prompt style,
    instead of sending the user to the separate panel. Proposing values is
    just a suggestion for a human to review; the actual grant still only
    ever happens via a discrete POST /api/authority/grant call triggered
    by an explicit user click - this model carries no authorization power
    by itself.
    """
    suggested_max_spend: float
    currency: str
    category: str
    duration_minutes: int
    reason: str


class CheckoutResponse(BaseModel):
    order: OrderOut
    guardian: GuardianResultOut
    payment_mode: Optional[str] = None  # "stub" | "real", only set when guardian allowed
    razorpay_order_id: Optional[str] = None
    razorpay_key_id: Optional[str] = None  # public key id, only in real mode
    amount_paise: Optional[int] = None
    authority_proposal: Optional[AuthorityProposalOut] = None


class PaymentConfigOut(BaseModel):
    mode: str  # "stub" | "real"
    key_id: Optional[str] = None
