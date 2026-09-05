"""
Idempotent DB init + seed. Safe to call on every backend startup - only
inserts data if the products table is empty.
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from .database import Base, engine, get_session
from .models import Merchant, Product
from .seed_data import MERCHANTS, PRODUCTS

logger = logging.getLogger("agent_buyer.seed")


def _migrate_existing_tables() -> None:
    """Base.metadata.create_all() only creates tables that don't exist yet
    - it never ALTERs an existing one. Since this project's SQLite DB file
    persists across restarts on your machine, a column added to a model
    after the DB already has that table (like `orders.authority_id`, added
    for Fully-Autonomous mode) needs a tiny explicit migration or every
    query touching that column breaks with "no such column". This is
    intentionally a manual, obvious, one-column-at-a-time list rather than
    a migration framework - proportionate to a local SQLite dev DB, not a
    production system."""
    inspector = inspect(engine)
    if "orders" not in inspector.get_table_names():
        return  # brand new DB - create_all() above already included the column
    columns = {col["name"] for col in inspector.get_columns("orders")}
    if "authority_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE orders ADD COLUMN authority_id INTEGER"))
        logger.info("Migrated orders table: added authority_id column.")


def init_and_seed() -> None:
    Base.metadata.create_all(bind=engine)
    _migrate_existing_tables()

    session = get_session()
    try:
        if session.query(Product).count() > 0:
            logger.info("Product catalog already seeded, skipping.")
            return

        merchants_by_name: dict[str, Merchant] = {}
        for m in MERCHANTS:
            merchant = Merchant(name=m["name"], description=m["description"])
            session.add(merchant)
            merchants_by_name[m["name"]] = merchant
        session.flush()  # assign IDs

        for p in PRODUCTS:
            merchant = merchants_by_name[p["merchant"]]
            product = Product(
                merchant_id=merchant.id,
                sku=p["sku"],
                name=p["name"],
                brand=p["brand"],
                category=p["category"],
                description=p["description"],
                price=p["price"],
                currency="INR",
                rating=p["rating"],
                review_count=p["review_count"],
                stock_qty=p["stock_qty"],
                attributes=p["attributes"],
            )
            session.add(product)

        session.commit()
        logger.info("Seeded %d merchants and %d products.", len(MERCHANTS), len(PRODUCTS))
    finally:
        session.close()
