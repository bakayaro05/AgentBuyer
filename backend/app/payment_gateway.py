"""
Payment gateway abstraction.

This is the ONLY module that talks to Razorpay (or, right now, simulates
it). The Payment Guardian decides ALLOW/DENY completely independently of
this module and BEFORE it is ever called - this file has no say in
whether a payment is authorized, it only executes one once the Guardian
has already approved it. That ordering is the whole point of the
architecture: LLM -> tool call -> Guardian -> ALLOW/DENY -> Razorpay,
never the reverse.

Two implementations behind one interface:
- StubPaymentGateway: no network calls, no real credentials. Generates
  fake order/payment ids so the full order lifecycle (SELECTED ->
  ORDER_CREATED -> PAYMENT_PENDING -> PAID/FAILED) can be exercised and
  demoed before you have real Razorpay Test Mode keys.
- RazorpayGateway: the real Razorpay Test Mode API via the official SDK.

`get_payment_gateway()` picks whichever one applies based on whether
RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are set in .env - no other code
needs to change when you add real keys later.
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from .config import settings

logger = logging.getLogger("agent_buyer.payment_gateway")


class PaymentGateway(ABC):
    @property
    @abstractmethod
    def mode(self) -> str:
        """'stub' or 'real' - surfaced to the frontend so it knows whether
        to render a real Razorpay Checkout widget or the demo simulate
        buttons."""
        raise NotImplementedError

    @abstractmethod
    def create_order(self, amount_paise: int, currency: str, receipt: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def verify_payment_signature(
        self, razorpay_order_id: str, razorpay_payment_id: str, razorpay_signature: str
    ) -> bool:
        raise NotImplementedError


class StubPaymentGateway(PaymentGateway):
    mode = "stub"

    def create_order(self, amount_paise: int, currency: str, receipt: str) -> dict[str, Any]:
        order_id = f"order_stub_{uuid.uuid4().hex[:14]}"
        logger.info(
            "[stub gateway] create_order amount=%s %s receipt=%s -> %s",
            amount_paise, currency, receipt, order_id,
        )
        return {
            "id": order_id,
            "amount": amount_paise,
            "currency": currency,
            "receipt": receipt,
            "status": "created",
        }

    def verify_payment_signature(
        self, razorpay_order_id: str, razorpay_payment_id: str, razorpay_signature: str
    ) -> bool:
        # There is no real signature in stub mode - the "signature" is
        # whatever the /simulate-payment endpoint constructed. Any
        # non-empty payment id counts as a valid simulated payment; a
        # simulated FAILURE never calls this at all (see orders.py).
        return bool(razorpay_payment_id)


class RazorpayGateway(PaymentGateway):
    mode = "real"

    def __init__(self, key_id: str, key_secret: str):
        import razorpay  # imported lazily so the package is only required in real mode

        self._client = razorpay.Client(auth=(key_id, key_secret))

    def create_order(self, amount_paise: int, currency: str, receipt: str) -> dict[str, Any]:
        return self._client.order.create(
            {"amount": amount_paise, "currency": currency, "receipt": receipt}
        )

    def verify_payment_signature(
        self, razorpay_order_id: str, razorpay_payment_id: str, razorpay_signature: str
    ) -> bool:
        try:
            self._client.utility.verify_payment_signature(
                {
                    "razorpay_order_id": razorpay_order_id,
                    "razorpay_payment_id": razorpay_payment_id,
                    "razorpay_signature": razorpay_signature,
                }
            )
            return True
        except Exception:
            logger.exception("Razorpay signature verification failed")
            return False


_gateway: PaymentGateway | None = None


def get_payment_gateway() -> PaymentGateway:
    global _gateway
    if _gateway is None:
        if settings.razorpay_key_id and settings.razorpay_key_secret:
            logger.info("Using real Razorpay gateway (Test Mode keys detected)")
            _gateway = RazorpayGateway(settings.razorpay_key_id, settings.razorpay_key_secret)
        else:
            logger.info("No Razorpay keys configured - using stub payment gateway")
            _gateway = StubPaymentGateway()
    return _gateway
