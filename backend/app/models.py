"""
ORM models for the simulated merchant catalog.

This is deliberately the ONLY place product/merchant facts are read from.
The LLM never invents prices, stock, or attributes - everything it sees
about a product comes from a row here, retrieved through the merchant
tool functions in `merchant_tools.py`.
"""
from __future__ import annotations

import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")

    products: Mapped[list["Product"]] = relationship(back_populates="merchant")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"))
    sku: Mapped[str] = mapped_column(String(40), unique=True)

    name: Mapped[str] = mapped_column(String(200))
    brand: Mapped[str] = mapped_column(String(80))
    category: Mapped[str] = mapped_column(String(80), index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    price: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="INR")

    rating: Mapped[float] = mapped_column(Float, default=0.0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    stock_qty: Mapped[int] = mapped_column(Integer, default=0)

    # Free-form category-specific facts, e.g. {"use_case": "marathon",
    # "cushioning": "high", "drop_mm": "8", "sizes_available": "7,8,9,10"}
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)

    merchant: Mapped["Merchant"] = relationship(back_populates="products")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sku": self.sku,
            "merchant": self.merchant.name if self.merchant else None,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "description": self.description,
            "price": self.price,
            "currency": self.currency,
            "rating": self.rating,
            "review_count": self.review_count,
            "in_stock": self.stock_qty > 0,
            "stock_qty": self.stock_qty,
            "attributes": self.attributes or {},
        }


class Order(Base):
    """An attempted purchase, from selection through payment.

    Deliberately denormalized (product name/price/merchant copied onto the
    order at selection time) rather than just holding a foreign key to
    Product - an order is a record of what was agreed to AT THAT MOMENT,
    which is exactly what the Payment Guardian needs to compare against
    the product's CURRENT state at checkout time (price changes, stock
    changes, etc). `events` is a lightweight audit trail for this specific
    order, separate from the recommendation-stage activity log.
    """
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)

    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    sku: Mapped[str] = mapped_column(String(40))
    product_name: Mapped[str] = mapped_column(String(200))
    merchant: Mapped[str] = mapped_column(String(120))

    unit_price_at_selection: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    quantity: Mapped[int] = mapped_column(Integer, default=1)

    # Set only when this order was ALLOWed under a Fully-Autonomous
    # delegated authority record - lets a successful payment later credit
    # its spend against that specific authority's budget (see
    # DelegatedAuthority below). Null for Semi-Autonomous orders (and any
    # pre-round-15 Recommendation-mode order still in the DB), which have
    # no authority to charge against.
    authority_id: Mapped[int | None] = mapped_column(
        ForeignKey("delegated_authority.id"), nullable=True
    )

    autonomy_mode: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="SELECTED")

    user_approved: Mapped[bool] = mapped_column(Boolean, default=False)

    guardian_allowed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    guardian_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    razorpay_order_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    razorpay_payment_id: Mapped[str | None] = mapped_column(String(80), nullable=True)

    events: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def log_event(self, step: str, detail: str) -> None:
        events = list(self.events or [])
        events.append(
            {
                "step": step,
                "detail": detail,
                "at": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        self.events = events

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sku": self.sku,
            "product_name": self.product_name,
            "merchant": self.merchant,
            "unit_price_at_selection": self.unit_price_at_selection,
            "currency": self.currency,
            "quantity": self.quantity,
            "autonomy_mode": self.autonomy_mode,
            "status": self.status,
            "user_approved": self.user_approved,
            "guardian_allowed": self.guardian_allowed,
            "guardian_reason": self.guardian_reason,
            "razorpay_order_id": self.razorpay_order_id,
            "razorpay_payment_id": self.razorpay_payment_id,
            "authority_id": self.authority_id,
            "events": self.events or [],
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
        }


class DelegatedAuthority(Base):
    """A user-granted spending authority for Fully-Autonomous mode.

    This is the actual authorization record the spec calls for - "max
    spend / allowed category / validity-expiry" - and it is the ONLY
    thing that can make the Guardian ALLOW a fully_autonomous purchase.
    Nothing about this record is ever created, modified, or extended by
    the LLM; only a human clicking "Grant authority" (or "Revoke") in the
    UI touches it, via the /api/authority/* endpoints. Granting a new one
    does not delete or edit an earlier record for the same session - the
    Guardian and the API always look at the MOST RECENT row, so granting
    fresh authority is how an expired or revoked one gets replaced, while
    the old row stays in place as part of the audit trail.
    """
    __tablename__ = "delegated_authority"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)

    max_spend: Mapped[float] = mapped_column(Float)
    spent: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    category: Mapped[str] = mapped_column(String(80))  # "any" or one of KNOWN_CATEGORIES

    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime)

    def to_dict(self) -> dict:
        now = datetime.datetime.utcnow()
        return {
            "id": self.id,
            "max_spend": self.max_spend,
            "spent": self.spent,
            "remaining": max(self.max_spend - self.spent, 0.0),
            "currency": self.currency,
            "category": self.category,
            "revoked": self.revoked,
            "expired": self.expires_at <= now,
            "active": (not self.revoked) and self.expires_at > now,
            "created_at": self.created_at.isoformat() + "Z",
            "expires_at": self.expires_at.isoformat() + "Z",
        }
