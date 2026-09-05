# Agent Buyer

**Secure Agentic Buyer — Razorpay Buildathon 2026.**

An AI agent that discovers, evaluates, and purchases products on a user's
behalf, built around one core principle: **the AI can act on your behalf,
but it can only spend money within authority you explicitly delegate to
it.** The centerpiece isn't the shopping chat — it's the authority and
guardrail architecture underneath it.

```
User → AI Agent → Tool/API call → Payment Guardian → ALLOW/DENY → Razorpay
```

The LLM requests an action. The backend — never the LLM — independently
decides whether that action is authorized. The LLM cannot modify its own
spending limits, approve its own purchase, or talk its way past the
Guardian; the Guardian's checks run against trusted backend state (the
Order row, a fresh DB read of the product, the delegated-authority
record), never against anything the LLM said.

## What it does

Tell it what you're shopping for in plain language:

> Find me good running shoes under ₹4,000 for marathon training

It extracts a structured shopping context (category, budget, keywords,
attributes), searches a simulated but realistic merchant catalog via
tool-calling, reasons over the real results, and recommends or acts —
depending on which of two **autonomy modes** it's in:

- **Semi-Autonomous** (the default) — searches, reasons, and shows you
  candidates; will only buy a specific product once you've explicitly
  approved that exact purchase.
- **Fully Autonomous** — you delegate authority up front (a max spend, a
  category, and an expiry window); within those limits, the agent
  searches, selects, checks out, and pays without asking again.

You never have to click a mode button — the agent reads your phrasing
("just buy the best one" vs. "let me approve first") and switches modes
itself, always announcing the switch in chat so it's never a surprise.

Before any payment, a **Payment Guardian** — a deterministic, independent
backend module the LLM has no access to — validates the order: does this
mode even permit a purchase, does the product still exist and have stock,
is the currency supported, has the price changed since you last saw it,
and (mode-specific) is there a real approval or a real, unexpired,
sufficient delegated authority behind this exact purchase. Only if every
check passes does the backend call Razorpay (Test Mode).

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running, with a tool-calling-
  capable model pulled. The project currently runs against
  `mistral:7b-instruct`; `llama3.1:8b` also works. Whichever you use, it
  must support Ollama's native tool-calling — the recommendation agent
  depends on it (see "Known limitations" below).

  ```powershell
  ollama pull mistral:7b-instruct
  ```

## Running it

**Backend:**

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Check it's alive at http://localhost:8000/api/health. If Ollama isn't
running or the model isn't pulled, `/api/chat` returns a clear error
instead of a generic 500. The product catalog (SQLite, `backend/agent_buyer.db`)
auto-creates and auto-seeds on first startup — 3 simulated merchants, 48
products across 6 categories (running shoes, casual sneakers, formal
shoes, backpacks, wireless earbuds, smartwatches). Sanity-check it directly
at http://localhost:8000/api/products (optionally `?category=running%20shoes`).

Config lives in `backend/.env` (copy from `.env.example`):
- `OLLAMA_HOST` / `OLLAMA_MODEL`
- `FRONTEND_ORIGIN` — CORS, defaults to the Vite dev server
- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` — leave both empty to run
  against a built-in stub gateway (good for demoing the Guardian's
  ALLOW/DENY logic without touching Razorpay at all); fill in real
  **Test Mode** keys (Razorpay Dashboard → Test Mode → Account & Settings
  → API Keys — no business verification needed) to exercise the real
  payment flow. **Never commit real key values to source control or
  share them in screenshots/recordings**, even Test Mode ones.

**Frontend**, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## The two autonomy modes

| Mode | Can search & recommend | Can buy | Requires |
|---|---|---|---|
| **Semi-Autonomous** (default) | Yes | Only the exact product you approve | Explicit approval per order |
| **Fully Autonomous** | Yes | Yes, automatically | An active, unexpired Delegated Authority covering this purchase's category and spend |

A third mode, **Recommendation** (search/recommend only, no purchase
capability at all), existed through round 14 and was removed in round 15
— see "Bugfix rounds" below for why.

**Delegated Authority** (Fully Autonomous only): a record the user grants
explicitly — max spend, allowed category, and an expiry window — that the
Payment Guardian checks independently on every purchase attempt in that
mode. It is never inferred from conversation; it's a real backend record
created via `/api/authority/grant` (and offered as an in-chat quick action
when a Fully-Autonomous purchase would otherwise be denied for lack of
one) and can be revoked at any time via `/api/authority/revoke`.

## The Payment Guardian

`backend/app/guardian.py` — the single most important module in the
project. Every check runs against trusted backend state; nothing the LLM
says can skip a step. In order:

1. **Autonomy mode allows purchasing at all** — an allowlist of the two
   purchase-capable modes; anything else (including a stale/unrecognized
   mode value) is DENY.
2. **Product still exists** in the catalog.
3. **Inventory** — still enough stock for the requested quantity.
4. **Currency** is one the system supports.
5. **Price revalidation** — the current catalog price is re-checked
   against what the user selected; a price increase between search and
   payment is caught here, in every mode, not just Fully-Autonomous.
6. **Mode-specific authorization**:
   - Semi-Autonomous: DENY unless the user has explicitly approved THIS
     exact order (not just selected/viewed it).
   - Fully Autonomous: DENY unless an active, unexpired Delegated
     Authority exists for this session AND this purchase's category and
     total are within what it covers.

Razorpay is never called before this returns ALLOW. `evaluate_iter()` is
the source of truth — a generator yielding each check the instant it's
decided, so the UI's live "thought stream" can show the Guardian's
reasoning as it happens rather than only after the fact.

## Security demonstrations (spec-required)

| # | Scenario | Status |
|---|---|---|
| 1 | Budget violation → DENY | Confirmed live |
| 2 | Category violation → DENY | Confirmed live |
| 3 | Expired delegated authority → DENY | Confirmed live |
| 4 | A mode with no purchasing capability attempts a purchase → DENY | Confirmed live pre-round-15 (via the old Recommendation mode). Since round 15 removed that mode from the UI, this now has to be demonstrated by sending an unrecognized/legacy mode value directly to the backend rather than clicking a mode button — the Guardian's allowlist check (guardian.py) still enforces it identically, just without a normal UI path to trigger it. **Decide before the final demo whether to keep this in the script this way, or drop it.** |
| 5 | Semi-Autonomous purchase attempted without approval → DENY | Confirmed live |
| 6 | Price increase after search → revalidated, DENY if over limit | Confirmed live |
| 7 | Prompt injection in malicious product metadata → Guardian still prevents unauthorized spending | Not yet exercised |

## What's built, phase by phase

- **Phase 1 — Understanding:** natural-language request → structured
  shopping context (category, budget, keywords, attributes), merged turn
  by turn.
- **Phase 2 — Simulated merchants + recommendations:** a seeded SQLite
  catalog behind an AI-readable tool surface (`search_products`,
  `get_product`, `check_inventory`); the LLM must call the tool and reason
  only over what it returns — it's instructed never to invent a product,
  and the backend forces a real search from context if the model skips
  the tool call, so recommendations are always grounded in real data.
- **Phase 3 — Semi-Autonomous + Payment Guardian + real Razorpay:** user
  approval flow, the Guardian described above, and real Razorpay Test
  Mode integration (falls back to a stub gateway with no keys configured).
- **Phase 4 — Fully Autonomous + Delegated Authority:** the authority
  grant/revoke flow and the Guardian's category/budget/expiry checks
  against it.
- **V2:** auto-search (no manual "Find products" click needed once the
  agent judges it has enough to go on), phrasing-based autonomy-mode
  inference (say "just buy it" or "let me approve first" instead of
  clicking a button — always announced in chat), a live "thought stream"
  showing each step (understanding → merging → searching → reasoning →
  Guardian checks) as it happens, and a decluttering pass on the
  candidates list.
- **V3:** an in-chat "Grant authority" quick action — offered the moment
  a Fully-Autonomous purchase is denied for lack of one, with concrete
  proposed values computed from the order itself, so granting authority
  never requires leaving the chat.

## Bugfix rounds

Fifteen rounds of live-testing and fixes so far, each triggered by an
actual observed failure (never a hypothetical). Rounds 1–13 covered:
budget-parsing edge cases (reversed ranges, floor-vs-ceiling
misreadings), category-vocabulary drift and cross-item context
contamination on a topic switch, autonomy-mode inference versatility (an
LLM-quoted "grounded evidence" signal plus an independent regex fallback,
made unconditional in round 13 so a bare "buy it" always works regardless
of what the model itself proposed), auto-search reliability (dedup so a
follow-up question doesn't re-trigger a pointless search), and
payment-outcome visibility (a chat bubble + banner on every resolved
order, not just a silent status change).

**Round 14** fixed two things reported together: (1) "buy it for me" was
silently doing nothing once *any* order existed in state, even an
unrelated one for a completely different, earlier item stuck in
`PAYMENT_PENDING` after a failed test payment — both of the mechanisms
that turn "buy it" into an actual purchase now only treat an order as
"blocking" when it's actually for one of the products currently on
screen. (2) A genuine product/category switch now resets the session's
autonomy mode back to Semi-Autonomous before that turn's own phrasing is
applied — so a leftover Fully-Autonomous stance from a finished, unrelated
item never silently carries over and auto-purchases something new (the
same message can still re-promote it, e.g. "I also want a sports watch,
just buy it").

**Round 15** removed the Recommendation autonomy mode entirely — reasoning
(confirmed against the code, not just in theory): Semi-Autonomous already
does everything Recommendation did, plus the ability to buy on approval,
so Recommendation was a strict, purposeless subset. Semi-Autonomous is now
the default a fresh session starts in and the only non-fully-autonomous
mode; the Guardian's mode gate became a defensive allowlist instead of a
named check (see security demo #4 above); the LLM prompt now maps
"just browsing" phrasing onto Semi-Autonomous rather than a no-op mode
that no longer exists.

## Known limitations

- **Single shopping context per session, not multiple concurrent items.**
  The system tracks one shopping context that refines turn by turn. A
  genuine topic switch (a new category the model actually extracts) resets
  it cleanly; if the model fails to extract a category for a second,
  unrelated item mentioned mid-conversation, its keywords/attributes can
  still merge into the wrong context. Out of scope for the MVP success
  criteria (all single-item flows) — worth revisiting for shopping-list-
  style multi-item support.
- **No real internet product discovery.** All merchants/products are
  simulated (seeded SQLite catalog) per the spec's MVP scope — finding a
  product online doesn't imply the agent can transact with that merchant,
  and that boundary was deliberately never blurred here.
- The recommendation agent depends on the local model actually using
  Ollama's native tool-calling; not all models are consistent about this
  (there's a backend fallback that forces a search from context if the
  model skips the tool call).
- `min_price` is rarely set even for an explicit range like "2000 to
  3000" — the prompt was deliberately tightened against misreading a
  plain budget as a floor, and now leans toward never setting one at all.
  Not a correctness risk, just a wider result set than strictly asked for.
- Security demo #4's UI path is gone as of round 15 (see the table above).
- Bugfix round 13 (mode-inference independence) and round 15 (this
  Recommendation-mode removal) are shipped but pending their own explicit
  live re-test as of this writing.

## API reference

`POST /api/chat`, `/api/chat/stream` — the understanding step (SSE
streaming variant included). `POST /api/recommend`, `/api/recommend/stream`
— search + reasoning. `GET /api/products`, `/api/categories` — catalog
introspection. `POST /api/session/mode` — manual mode override.
`POST /api/authority/grant`, `GET /api/authority`, `POST
/api/authority/revoke` — Delegated Authority lifecycle. `POST
/api/orders/select` → `.../approve` → `.../checkout` (+ `/stream` variant)
→ `.../simulate-payment` or `.../verify-payment` — the order lifecycle;
`.../payment-attempt-failed` records a failed attempt without failing the
whole order. `GET /api/orders/{id}`, `/api/orders` — order/audit lookup.
`GET /api/payment-config` — which gateway (real Razorpay vs. stub) is
active. `POST /api/reset`, `/api/session/new`, `GET /api/session/{id}` —
session lifecycle.

## Architecture note

The LLM only ever produces structured JSON (intent, desired mode, replies)
and reasons over tool results it's handed — it has no access to real
purchasing capability at any point. `backend/app/llm_client.py` is written
as a swappable interface (`LLMClient`) specifically so a cloud model
(Anthropic/OpenAI) can replace Ollama later without touching the agent or
route logic in `app/main.py`. The order lifecycle
(`SEARCHED → SELECTED → ORDER_CREATED → PAYMENT_PENDING → PAID / FAILED /
CANCELLED`) and every step the agent takes are logged and exposed to the
UI's Activity Log, per the spec's auditability requirement.
</content>
