import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import "./App.css";
import { ThoughtStream } from "./ThoughtStream";
import {
  approveOrder,
  getAuthority,
  getCategories,
  getPaymentConfig,
  grantAuthority,
  newSession,
  reportPaymentAttemptFailed,
  resetSession,
  revokeAuthority,
  selectOrder,
  setMode,
  simulatePayment,
  streamChat,
  streamCheckout,
  streamRecommend,
  verifyPayment,
} from "./api";
import type {
  AuditEntry,
  AuthorityProposal,
  AutonomyMode,
  DelegatedAuthority,
  GuardianResult,
  Order,
  PaymentConfig,
  Product,
  ShoppingContext,
  ThoughtStep,
} from "./types";

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Razorpay: new (options: Record<string, unknown>) => {
      open: () => void;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      on?: (event: string, handler: (response: any) => void) => void;
    };
  }
}

const EMPTY_CONTEXT: ShoppingContext = {
  category: null,
  keywords: [],
  min_price: null,
  max_price: null,
  currency: "INR",
  attributes: {},
  notes: null,
};

// V2 "thought stream": the chat feed now holds two kinds of entries -
// ordinary message bubbles, and a live, animated thought-stream block per
// agent turn (context extraction, product search, or a Guardian gate),
// so the agent's reasoning shows up inline exactly where it happened,
// the way Claude/ChatGPT/Grok show a "thinking" block above the reply.
// V3: an in-chat "Grant authority" quick action, in the same spirit as
// Claude's own tool-permission prompts - the agent (backend, never the
// LLM - see main.py's `_authority_proposal`) proposes concrete grant
// values right where the denial happened, and a real button click is
// still required to actually authorize anything; nothing here can grant
// itself. `status` tracks the human's response to THIS specific prompt.
type Entry =
  | { kind: "message"; role: "user" | "assistant"; content: string }
  | { kind: "thought"; turnId: string; steps: ThoughtStep[]; active: boolean }
  | {
      kind: "authority_proposal";
      id: string;
      orderId: number;
      proposal: AuthorityProposal;
      status: "pending" | "granting" | "granted" | "dismissed" | "error";
      error?: string;
    };

// Decluttering pass: only this many candidate cards show by default; the
// rest are one "Show more" click away.
const CANDIDATE_PREVIEW_COUNT = 3;

// Bugfix round 15: Recommendation mode removed - Semi-Autonomous already
// did everything it did (search, reason, show candidates) plus buy-on-
// approval, so it was a strict subset with no purpose of its own. This
// map now has only two entries, which also shrinks the mode-button row
// below (it renders one button per key here) with no separate UI change
// needed.
const MODE_LABELS: Record<AutonomyMode, string> = {
  semi_autonomous: "Semi-Autonomous",
  fully_autonomous: "Fully Autonomous",
};

// Round 7: the user found the auto-detected mode switch hard to follow -
// specifically, a plain "I want to buy it" jumping straight to Fully
// Autonomous read as unclear/unexpected, and they couldn't tell from the
// UI what phrasing would have landed them in Semi-Autonomous instead. Per
// the user's explicit choice, the underlying classification rule (backend
// `_FULLY_AUTONOMOUS_SIGNAL`/`_SEMI_AUTONOMOUS_SIGNAL` in main.py) is
// unchanged - any purchase-imperative wording ("buy it", "I want to buy
// it") still means Fully Autonomous, and only explicit "show/approve
// before buying" phrasing means Semi-Autonomous. What changes here is
// legibility: the in-chat announcement now says WHY that mode was picked
// and what to say instead for the other mode, so the distinction is
// visible in the moment rather than something you have to infer.
const MODE_SWITCH_WHY: Record<AutonomyMode, string> = {
  semi_autonomous: "you asked to review or approve before anything is bought",
  fully_autonomous: "you gave a purchase instruction with no request to review first",
};
const MODE_SWITCH_WHAT_HAPPENS: Partial<Record<AutonomyMode, string>> = {
  semi_autonomous: "I'll show you the product and wait for your explicit approval before paying.",
  fully_autonomous: "I'll select, run the Payment Guardian, and complete payment without asking again (still subject to delegated authority).",
};

function formatPrice(value: number | null, currency: string) {
  if (value === null) return null;
  return `${currency} ${value.toLocaleString("en-IN")}`;
}

function ContextPanel({
  title,
  subtitle,
  context,
}: {
  title: string;
  subtitle: string;
  context: ShoppingContext;
}) {
  const isEmpty =
    !context.category &&
    context.keywords.length === 0 &&
    context.min_price === null &&
    context.max_price === null &&
    Object.keys(context.attributes).length === 0 &&
    !context.notes;

  return (
    <div className="context-card">
      <h2>{title}</h2>
      <p className="sub">{subtitle}</p>
      {isEmpty ? (
        <p className="empty">Nothing extracted yet.</p>
      ) : (
        <>
          <div className="context-row">
            <span className="k">Category</span>
            <span className="v">{context.category ?? "—"}</span>
          </div>
          <div className="context-row">
            <span className="k">Budget</span>
            <span className="v">
              {context.min_price === null && context.max_price === null
                ? "—"
                : `${formatPrice(context.min_price, context.currency) ?? "any"} – ${
                    formatPrice(context.max_price, context.currency) ?? "any"
                  }`}
            </span>
          </div>
          <div className="context-row">
            <span className="k">Keywords</span>
            <span className="v">
              {context.keywords.length === 0 ? (
                "—"
              ) : (
                <span className="chips">
                  {context.keywords.map((kw) => (
                    <span className="chip" key={kw}>
                      {kw}
                    </span>
                  ))}
                </span>
              )}
            </span>
          </div>
          <div className="context-row">
            <span className="k">Attributes</span>
            <span className="v">
              {Object.keys(context.attributes).length === 0 ? (
                "—"
              ) : (
                <span className="chips">
                  {Object.entries(context.attributes).map(([k, v]) => (
                    <span className="chip" key={k}>
                      {k}: {v}
                    </span>
                  ))}
                </span>
              )}
            </span>
          </div>
          {context.notes && (
            <div className="context-row">
              <span className="k">Notes</span>
              <span className="v">{context.notes}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function AuditLog({ entries }: { entries: AuditEntry[] }) {
  return (
    <div className="audit-log">
      {entries.map((e, i) => (
        <div className="audit-row" key={i}>
          <span className="audit-check">✓</span>
          <div>
            <div className="audit-step">{e.step}</div>
            <div className="audit-detail">{e.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Decluttering pass: the ALLOW/DENY headline - the actual visible proof
// that the backend, not the LLM, decided this - always stays on screen
// unconditionally (never hidden, per the project's own architectural
// thesis); only the individual check-by-check breakdown is collapsed by
// default, one click away, same "collapsed by default, expand for detail"
// treatment as the chat's thought stream and the side-panel cards below.
function GuardianPanel({ result }: { result: GuardianResult }) {
  const [collapsed, setCollapsed] = useState(true);
  const failCount = result.checks.filter((c) => !c.passed).length;
  return (
    <div className={`guardian-verdict ${result.allowed ? "allow" : "deny"}`}>
      <button
        type="button"
        className="guardian-headline-btn"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        <span className="guardian-headline">
          {result.allowed ? "✓ ALLOW" : "✕ DENY"} — {result.reason}
        </span>
        <span className="collapsible-summary">
          {result.checks.length} check{result.checks.length === 1 ? "" : "s"}
          {failCount > 0 ? ` · ${failCount} failed` : ""}
        </span>
        <span className="collapsible-caret">{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && (
        <div className="guardian-checks">
          {result.checks.map((c, i) => (
            <div className={`guardian-check-row ${c.passed ? "pass" : "fail"}`} key={i}>
              <span className="gc-mark">{c.passed ? "✓" : "✕"}</span>
              <div>
                <div className="gc-name">{c.name}</div>
                <div className="gc-detail">{c.detail}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Generic collapsed-by-default section, reused for the Activity log,
// Order history, and (variant="card") any side-panel card whose detail
// isn't needed at a glance - same visual language as the chat's own
// thought-stream ("Thought process", collapsed, click to expand).
function Collapsible({
  title,
  summary,
  defaultCollapsed = true,
  variant = "card",
  children,
}: {
  title: string;
  summary?: string;
  defaultCollapsed?: boolean;
  variant?: "card" | "inline";
  children: ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  return (
    <div className={variant === "card" ? "context-card collapsible-card" : "collapsible-inline"}>
      <button
        type="button"
        className="collapsible-header"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        {variant === "card" ? <h2>{title}</h2> : <span className="collapsible-title">{title}</span>}
        {summary && <span className="collapsible-summary">{summary}</span>}
        <span className="collapsible-caret">{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && <div className="collapsible-body">{children}</div>}
    </div>
  );
}

function OrderEventLog({ order }: { order: Order }) {
  return (
    <div className="audit-log">
      {order.events.map((e, i) => (
        <div className="audit-row" key={i}>
          <span className="audit-check">•</span>
          <div>
            <div className="audit-step">{e.step}</div>
            <div className="audit-detail">{e.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function statusClass(status: string): string {
  if (status === "PAID") return "status-paid";
  if (status === "FAILED") return "status-failed";
  if (status === "PAYMENT_PENDING") return "status-pending";
  return "status-selected";
}

function ProductCard({
  product,
  recommended,
  onBuy,
  buying,
}: {
  product: Product;
  recommended: boolean;
  onBuy: (sku: string) => void;
  buying: boolean;
}) {
  return (
    <div className={`product-card ${recommended ? "recommended" : ""}`}>
      {recommended && <div className="badge">Recommended</div>}
      <div className="product-top">
        <div>
          <div className="product-name">{product.name}</div>
          <div className="product-brand">
            {product.brand} · {product.merchant}
          </div>
        </div>
        <div className="product-price">{formatPrice(product.price, product.currency)}</div>
      </div>
      <p className="product-desc">{product.description}</p>
      <div className="chips">
        {Object.entries(product.attributes)
          .slice(0, 4)
          .map(([k, v]) => (
            <span className="chip" key={k}>
              {k}: {v}
            </span>
          ))}
      </div>
      <div className="product-meta">
        <span>★ {product.rating.toFixed(1)} ({product.review_count})</span>
        <span className={product.in_stock ? "in-stock" : "out-stock"}>
          {product.in_stock ? `In stock (${product.stock_qty})` : "Out of stock"}
        </span>
      </div>
      <button
        className="buy-btn"
        onClick={() => onBuy(product.sku)}
        disabled={buying || !product.in_stock}
      >
        {buying ? "…" : "Buy this"}
      </button>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [turnIntent, setTurnIntent] = useState<ShoppingContext>(EMPTY_CONTEXT);
  const [context, setContext] = useState<ShoppingContext>(EMPTY_CONTEXT);

  const [recommending, setRecommending] = useState(false);
  const [recommendError, setRecommendError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<Product[]>([]);
  const [recommendedSkus, setRecommendedSkus] = useState<string[]>([]);
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([]);
  // Decluttering pass: don't dump every candidate on screen at once - show
  // a short preview and let the user expand for the rest.
  const [showAllCandidates, setShowAllCandidates] = useState(false);

  // Bugfix round 15: a fresh session now starts in Semi-Autonomous, not
  // the removed Recommendation mode - matches `_get_session`'s new
  // default on the backend.
  const [autonomyMode, setAutonomyMode] = useState<AutonomyMode>("semi_autonomous");
  const [paymentConfig, setPaymentConfig] = useState<PaymentConfig | null>(null);
  const [order, setOrder] = useState<Order | null>(null);
  const [guardianResult, setGuardianResult] = useState<GuardianResult | null>(null);
  const [orderBusy, setOrderBusy] = useState(false);
  const [orderError, setOrderError] = useState<string | null>(null);
  const [buyingSku, setBuyingSku] = useState<string | null>(null);
  const [checkoutInfo, setCheckoutInfo] = useState<
    { amountPaise: number; razorpayOrderId: string; razorpayKeyId: string } | null
  >(null);
  const [payingNow, setPayingNow] = useState(false);
  // Bugfix round 9 (issue 2): a real/simulated payment outcome previously
  // only ever changed the Order panel's status badge - nothing in the chat
  // feed or anywhere else on screen announced it, so a PAID/FAILED result
  // was easy to miss unless you were specifically looking at that one card
  // (confirmed by the user: "it shows Paid in the order history but not
  // anywhere else in the entire UI"). This banner is a second, more visible
  // channel: shown right under the header for a few seconds on every
  // payment outcome (success or failure, real or simulated), alongside a
  // permanent chat message (see showPaymentOutcome below) that stays in
  // the conversation history after the banner itself has faded.
  const [paymentBanner, setPaymentBanner] = useState<{ kind: "success" | "failure"; text: string } | null>(null);

  const [categories, setCategories] = useState<string[]>([]);
  const [authority, setAuthority] = useState<DelegatedAuthority | null>(null);
  const [authorityBusy, setAuthorityBusy] = useState(false);
  const [authorityError, setAuthorityError] = useState<string | null>(null);
  const [grantMaxSpend, setGrantMaxSpend] = useState("5000");
  const [grantCategory, setGrantCategory] = useState("any");
  const [grantDurationMinutes, setGrantDurationMinutes] = useState("60");

  const scrollRef = useRef<HTMLDivElement>(null);
  const autoFlowRef = useRef<{
    boughtSku: string | null;
    checkedOutOrderId: number | null;
    paidOrderId: number | null;
    deniedOrderId: number | null;
  }>({
    boughtSku: null,
    checkedOutOrderId: null,
    paidOrderId: null,
    deniedOrderId: null,
  });
  // Pending setTimeouts for the narrated auto-flow below. Kept in a ref
  // (not an effect cleanup) so an unrelated re-render of the scheduling
  // effect can't cancel a step that's already been committed to - only
  // an explicit reset/mode-switch or unmount clears these.
  const autoFlowTimersRef = useRef<number[]>([]);
  const AUTO_FLOW_DELAY_MS = 3000;
  // V2 Batch 2: shorter than the fully-autonomous narration delay above -
  // the assistant's own reply already serves as the "searching now"
  // announcement (the LLM is prompted to phrase it that way when
  // ready_to_search is true), so this just gives the user a beat to read
  // that reply before the search actually fires.
  const AUTO_SEARCH_DELAY_MS = 900;
  // V2 Batch 2 bugfix: the local model's `ready_to_search` flag is decided
  // fresh each turn and doesn't know whether we already searched with this
  // exact context - so a pure follow-up question about an already-found
  // product (e.g. "is it good?") can come back `ready_to_search: true`
  // again (the context still has a category/budget from earlier turns)
  // and trigger a pointless repeat search instead of answering the
  // question. Tracked here instead: only actually fire the auto-search
  // when the accumulated context has materially changed since the last
  // time we auto-searched with it - a real refinement (new budget,
  // attribute, category) always changes this JSON, while a follow-up
  // question that adds nothing new does not.
  const lastAutoSearchContextRef = useRef<string | null>(null);
  // Bugfix round 11: "buy it" / "I want you to buy it" said AFTER
  // candidates are already on screen (but before anything's been
  // selected) was falling through to `ready_to_search` again instead of
  // actually buying anything - the backend has no memory of candidates to
  // act on, so it can only report the raw "this message is a purchase
  // confirmation" signal (`buy_confirmation`) and leave the decision to
  // the frontend, which is the only place that knows what's currently
  // recommended. Same dedup shape as `lastAutoSearchContextRef` above, so
  // repeating "buy it" against the same unchanged context doesn't
  // re-select the same product over and over.
  const lastAutoBuyContextRef = useRef<string | null>(null);
  const paymentBannerTimerRef = useRef<number | null>(null);
  const PAYMENT_BANNER_DURATION_MS = 8000;

  function scheduleAutoFlowStep(fn: () => void, delayMs: number = AUTO_FLOW_DELAY_MS) {
    const id = window.setTimeout(fn, delayMs);
    autoFlowTimersRef.current.push(id);
  }

  function clearAutoFlowTimers() {
    autoFlowTimersRef.current.forEach((id) => window.clearTimeout(id));
    autoFlowTimersRef.current = [];
  }

  // --- V2 thought-stream helpers -----------------------------------
  // addMessage appends an ordinary chat bubble. startThought/upsertThoughtStep/
  // finishThought manage one live thought-stream block (identified by
  // turnId) inline in the same entries list - upsert matches on step id
  // so a "pending" step (e.g. Guardian check in flight) can be updated to
  // "done"/"error" in place rather than appearing twice.
  function addMessage(role: "user" | "assistant", content: string) {
    setEntries((prev) => [...prev, { kind: "message", role, content }]);
  }

  // Bugfix round 9 (issue 2): announces a payment outcome in TWO places at
  // once - a permanent chat bubble (via addMessage, so it stays in the
  // conversation history like every other narrated step) and a temporary
  // banner right under the header (so it's visible even if you're not
  // looking at the chat feed or the Order panel at that exact moment).
  // Called from every path that can resolve an order to PAID/FAILED: the
  // real-gateway verifyPayment handler, the payment.failed listener, and
  // both simulate-payment outcomes.
  function showPaymentOutcome(kind: "success" | "failure", text: string) {
    addMessage("assistant", (kind === "success" ? "✅ " : "❌ ") + text);
    setPaymentBanner({ kind, text });
    if (paymentBannerTimerRef.current !== null) {
      window.clearTimeout(paymentBannerTimerRef.current);
    }
    paymentBannerTimerRef.current = window.setTimeout(() => {
      setPaymentBanner(null);
      paymentBannerTimerRef.current = null;
    }, PAYMENT_BANNER_DURATION_MS);
  }

  function startThought(turnId: string) {
    setEntries((prev) => [...prev, { kind: "thought", turnId, steps: [], active: true }]);
  }

  function upsertThoughtStep(turnId: string, evt: { id?: string; label?: string; detail?: string; status?: string }) {
    if (!evt.id) return;
    setEntries((prev) =>
      prev.map((e) => {
        if (e.kind !== "thought" || e.turnId !== turnId) return e;
        const steps = [...e.steps];
        const idx = steps.findIndex((s) => s.id === evt.id);
        const nextStep: ThoughtStep = {
          id: evt.id!,
          label: evt.label ?? steps[idx]?.label ?? "",
          detail: evt.detail ?? steps[idx]?.detail,
          status: (evt.status as ThoughtStep["status"]) ?? "done",
        };
        if (idx >= 0) steps[idx] = nextStep;
        else steps.push(nextStep);
        return { ...e, steps };
      }),
    );
  }

  function finishThought(turnId: string) {
    setEntries((prev) =>
      prev.map((e) => (e.kind === "thought" && e.turnId === turnId ? { ...e, active: false } : e)),
    );
  }

  // --- V3 in-chat "Grant authority" quick action --------------------
  function addAuthorityProposal(orderId: number, proposal: AuthorityProposal): string {
    const id = `authority-proposal-${orderId}-${Date.now()}`;
    setEntries((prev) => [...prev, { kind: "authority_proposal", id, orderId, proposal, status: "pending" }]);
    return id;
  }

  function updateAuthorityProposal(id: string, patch: Partial<Extract<Entry, { kind: "authority_proposal" }>>) {
    setEntries((prev) =>
      prev.map((e) => (e.kind === "authority_proposal" && e.id === id ? { ...e, ...patch } : e)),
    );
  }

  const canRecommend = Boolean(context.category || context.keywords.length > 0);

  // Bugfix round 14: both the fully-autonomous auto-flow effect and the
  // round-11 direct-select path used to treat ANY existing `order` as "an
  // item is already in flight, don't auto-buy" - even when that order was
  // for a completely different, earlier item (e.g. a running-shoe order
  // still sitting in PAYMENT_PENDING after a failed Razorpay attempt) and
  // the user has since pivoted to a new item (a smartwatch) that just
  // returned fresh candidates. Because `order` state isn't cleared on a
  // topic switch, that stale unrelated order silently blocked the new
  // item from ever being auto-bought - "buy it for me" would just sit
  // there with no visible error. An order should only count as "blocking"
  // when it's actually for one of the products currently on screen (i.e.
  // still the same item); an order for a SKU that isn't among the current
  // candidates is for a different, earlier item and should never stop a
  // new one from being bought. `order.status` PAID/FAILED is also
  // terminal either way, so it never blocks regardless of SKU.
  const orderBlocksNewPurchase = Boolean(
    order &&
      order.status !== "PAID" &&
      order.status !== "FAILED" &&
      candidates.some((p) => p.sku === order.sku),
  );

  useEffect(() => {
    newSession().then(setSessionId).catch((e) => setError(String(e)));
    getPaymentConfig().then(setPaymentConfig).catch(() => setPaymentConfig(null));
    getCategories().then(setCategories).catch(() => setCategories([]));
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    getAuthority(sessionId).then(setAuthority).catch(() => setAuthority(null));
  }, [sessionId]);

  useEffect(() => clearAutoFlowTimers, []);

  // Fully-autonomous mode: the agent selects the top recommendation,
  // runs checkout (Payment Guardian), and completes payment without
  // requiring a manual click at any of those three steps - matching the
  // spec's criterion C ("delegated authority -> search -> selection ->
  // Payment Guardian -> Razorpay -> verified order", no approval step).
  // Each stage is guarded by a ref so it fires at most once per
  // candidate set / order id, a short delay + chat narration gives the
  // user time to actually read the pick/verdict before the next stage
  // fires, and a Guardian DENY (order.status becomes "FAILED") stops the
  // chain instead of retrying forever.
  useEffect(() => {
    if (autonomyMode !== "fully_autonomous") return;
    // Bugfix round 14: `orderBlocksNewPurchase` (not a blanket `|| order`)
    // - a stale order for a previous, different item must never stop this
    // item's candidates from being auto-bought. See its definition above.
    if (candidates.length === 0 || orderBlocksNewPurchase) return;
    const topSku = recommendedSkus[0] ?? candidates[0]?.sku;
    if (!topSku || autoFlowRef.current.boughtSku === topSku) return;
    autoFlowRef.current.boughtSku = topSku;
    const product = candidates.find((p) => p.sku === topSku);
    if (product) {
      addMessage(
        "assistant",
        `Fully autonomous mode: I'm going with ${product.name} (${formatPrice(
          product.price,
          product.currency
        )}) — ${product.description} Selecting it and running the Payment Guardian next.`,
      );
    }
    scheduleAutoFlowStep(() => handleBuy(topSku));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autonomyMode, candidates, recommendedSkus, order, orderBlocksNewPurchase]);

  useEffect(() => {
    if (autonomyMode !== "fully_autonomous") return;
    if (!order || order.status !== "SELECTED") return;
    if (autoFlowRef.current.checkedOutOrderId === order.id) return;
    autoFlowRef.current.checkedOutOrderId = order.id;
    addMessage(
      "assistant",
      `Order #${order.id} created for ${order.product_name} at ${formatPrice(
        order.unit_price_at_selection,
        order.currency
      )} — running the Payment Guardian now.`,
    );
    scheduleAutoFlowStep(() => handleCheckout());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autonomyMode, order]);

  useEffect(() => {
    if (autonomyMode !== "fully_autonomous") return;
    if (!order || order.status !== "PAYMENT_PENDING") return;
    if (autoFlowRef.current.paidOrderId === order.id) return;
    if (paymentConfig?.mode === "real" && !checkoutInfo) return;
    autoFlowRef.current.paidOrderId = order.id;
    addMessage("assistant", "Payment Guardian approved the order — completing payment now.");
    if (paymentConfig?.mode === "real") {
      scheduleAutoFlowStep(() => handlePayNow());
    } else if (paymentConfig?.mode === "stub") {
      scheduleAutoFlowStep(() => handleSimulate("success"));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autonomyMode, order, checkoutInfo, paymentConfig]);

  // Explain a Guardian DENY in the chat too, not just the verdict panel.
  useEffect(() => {
    if (autonomyMode !== "fully_autonomous") return;
    if (!order || order.status !== "FAILED" || !guardianResult) return;
    if (autoFlowRef.current.deniedOrderId === order.id) return;
    autoFlowRef.current.deniedOrderId = order.id;
    addMessage("assistant", `Payment Guardian denied this purchase: ${guardianResult.reason}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autonomyMode, order, guardianResult]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [entries]);

  async function handleSend() {
    const text = input.trim();
    if (!text || !sessionId || sending) return;
    setError(null);
    setInput("");
    addMessage("user", text);
    setSending(true);
    const turnId = `chat-${Date.now()}`;
    startThought(turnId);
    try {
      await streamChat(sessionId, text, (evt) => {
        if (evt.type === "step") {
          upsertThoughtStep(turnId, evt);
        } else if (evt.type === "result") {
          finishThought(turnId);
          addMessage("assistant", evt.reply as string);
          setTurnIntent(evt.turn_intent as ShoppingContext);
          setContext(evt.context as ShoppingContext);

          // V2 Batch 2 - mode inference: the backend already applied the
          // switch server-side (it's the same session the Guardian reads
          // from); mirror it into local UI state and announce it as its
          // own chat message, separate from the model's own reply, so a
          // misread mode is visible and correctable in the same turn
          // rather than discovered later.
          const newMode = evt.mode_changed as AutonomyMode | null | undefined;
          if (newMode) {
            setAutonomyMode(newMode);
            clearAutoFlowTimers();
            autoFlowRef.current = {
              boughtSku: null,
              checkedOutOrderId: null,
              paidOrderId: null,
              deniedOrderId: null,
            };
            addMessage(
              "assistant",
              `(Switched to ${MODE_LABELS[newMode]} mode — ${MODE_SWITCH_WHY[newMode]}. ` +
                `${MODE_SWITCH_WHAT_HAPPENS[newMode] ?? ""} ` +
                `Say "let me approve first" for Semi-Autonomous or "buy it" for Fully ` +
                `Autonomous any time, or click a mode button above to override.)`,
            );
          }

          // V2 Batch 2 - auto-search: no "Find products" click needed once
          // the agent judges it has enough to search (or the user said so
          // explicitly) - the model's reply text is itself the "searching
          // now" announcement, this just fires the actual search shortly
          // after so the reply is visible on screen first.
          if (evt.ready_to_search) {
            const contextSignature = JSON.stringify(evt.context ?? {});
            if (lastAutoSearchContextRef.current !== contextSignature) {
              lastAutoSearchContextRef.current = contextSignature;
              scheduleAutoFlowStep(() => handleRecommend(), AUTO_SEARCH_DELAY_MS);
            }
          }

          // Bugfix round 11: "buy it" said once a recommendation is
          // already on screen, with no order yet for it (nothing selected
          // yet, or the previous order for a different item already
          // resolved to PAID/FAILED) - select the top recommended
          // candidate, the same action "Buy this" performs, instead of
          // silently doing nothing while the reply claims to search again.
          // Deliberately skipped when an order is already SELECTED or
          // PAYMENT_PENDING for the current item - the backend's
          // order-in-progress guard already handles that case with its
          // own nudge reply pointing at the existing order.
          //
          // Bugfix round 12: also deliberately skipped whenever the
          // EFFECTIVE mode for this turn (the mode we're switching to
          // this turn, or the mode already in effect) is
          // fully_autonomous. That mode already has its own long-standing
          // auto-flow `useEffect` (further down) which fires the instant
          // fresh candidates exist with no order and the mode is
          // fully_autonomous - it re-fires on a mode switch too, since
          // `autonomyMode` is in its dependency array - and it does a
          // strictly better job: it posts its own "I'm going with X..."
          // narration before selecting, and its own ref-based guard
          // dedupes independently of this one. Letting BOTH fire at once
          // was exactly the confusing double-select the user hit live: a
          // "buy it" that ALSO switches to Fully Autonomous should look
          // identical to how it always has in that mode (mode switch ->
          // the existing chain quietly takes over) - never a separate,
          // narration-less "select" from here first.
          // Bugfix round 14: same `orderBlocksNewPurchase` fix as the
          // fully-autonomous effect above - a stale order for an earlier,
          // different item (not among the current `candidates`) must not
          // block buying the item actually on screen now.
          const effectiveMode = (newMode ?? autonomyMode) as AutonomyMode;
          if (
            evt.buy_confirmation &&
            effectiveMode !== "fully_autonomous" &&
            !orderBlocksNewPurchase &&
            candidates.length > 0 &&
            recommendedSkus.length > 0
          ) {
            const buySignature = JSON.stringify(evt.context ?? {});
            if (lastAutoBuyContextRef.current !== buySignature) {
              lastAutoBuyContextRef.current = buySignature;
              const topSku = recommendedSkus[0];
              scheduleAutoFlowStep(() => handleBuy(topSku), AUTO_SEARCH_DELAY_MS);
            }
          }
        } else if (evt.type === "error") {
          finishThought(turnId);
          setError(evt.message ?? "Something went wrong understanding that message.");
        }
      });
    } catch (e) {
      finishThought(turnId);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  async function handleRecommend() {
    if (!sessionId || recommending) return;
    setRecommendError(null);
    setRecommending(true);
    const turnId = `recommend-${Date.now()}`;
    startThought(turnId);
    try {
      await streamRecommend(sessionId, (evt) => {
        if (evt.type === "step") {
          upsertThoughtStep(turnId, evt);
        } else if (evt.type === "result") {
          finishThought(turnId);
          setCandidates((evt.candidates as Product[]) ?? []);
          setShowAllCandidates(false);
          setRecommendedSkus((evt.recommended_skus as string[]) ?? []);
          setAuditLog((evt.audit_log as AuditEntry[]) ?? []);
          addMessage("assistant", evt.reply as string);
        } else if (evt.type === "error") {
          finishThought(turnId);
          setRecommendError(evt.message ?? "Something went wrong searching for products.");
        }
      });
    } catch (e) {
      finishThought(turnId);
      setRecommendError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecommending(false);
    }
  }

  async function handleSetMode(mode: AutonomyMode) {
    if (!sessionId) return;
    setAutonomyMode(mode);
    clearAutoFlowTimers();
    autoFlowRef.current = { boughtSku: null, checkedOutOrderId: null, paidOrderId: null, deniedOrderId: null };
    try {
      await setMode(sessionId, mode);
    } catch (e) {
      setOrderError(e instanceof Error ? e.message : String(e));
    }
  }

  async function handleGrantAuthority() {
    if (!sessionId || authorityBusy) return;
    const maxSpend = Number(grantMaxSpend);
    const durationMinutes = Number(grantDurationMinutes);
    if (!Number.isFinite(maxSpend) || maxSpend <= 0) {
      setAuthorityError("Enter a max spend greater than 0.");
      return;
    }
    if (!Number.isFinite(durationMinutes) || durationMinutes <= 0) {
      setAuthorityError("Enter a valid duration in minutes.");
      return;
    }
    setAuthorityError(null);
    setAuthorityBusy(true);
    try {
      const granted = await grantAuthority(sessionId, {
        max_spend: maxSpend,
        category: grantCategory,
        duration_minutes: durationMinutes,
      });
      setAuthority(granted);
    } catch (e) {
      setAuthorityError(e instanceof Error ? e.message : String(e));
    } finally {
      setAuthorityBusy(false);
    }
  }

  async function handleRevokeAuthority() {
    if (!sessionId || authorityBusy) return;
    setAuthorityError(null);
    setAuthorityBusy(true);
    try {
      const revoked = await revokeAuthority(sessionId);
      setAuthority(revoked);
    } catch (e) {
      setAuthorityError(e instanceof Error ? e.message : String(e));
    } finally {
      setAuthorityBusy(false);
    }
  }

  // V3: the human-click half of the in-chat "Grant authority" quick
  // action - same Claude-permission-prompt shape as `handleGrantAuthority`
  // above (a real button click calling the exact same POST
  // /api/authority/grant), just triggered from the chat bubble instead of
  // the panel form, using the backend-computed suggested values instead
  // of whatever's currently in the panel's inputs. On success, retries
  // the SAME order's checkout automatically (only if it's still the
  // order in view) - the whole point of asking in-line is that this only
  // needed one more click to bring the same purchase to completion.
  async function handleGrantFromProposal(entryId: string, orderId: number, proposal: AuthorityProposal) {
    if (!sessionId) return;
    updateAuthorityProposal(entryId, { status: "granting" });
    try {
      const granted = await grantAuthority(sessionId, {
        max_spend: proposal.suggested_max_spend,
        category: proposal.category,
        duration_minutes: proposal.duration_minutes,
      });
      setAuthority(granted);
      updateAuthorityProposal(entryId, { status: "granted" });
      addMessage(
        "assistant",
        `Delegated authority granted: up to ${formatPrice(proposal.suggested_max_spend, proposal.currency)} ` +
          `for ${proposal.category === "any" ? "any category" : proposal.category}, valid ${proposal.duration_minutes} minutes. ` +
          "Retrying this purchase now.",
      );
      if (order && order.id === orderId) {
        scheduleAutoFlowStep(() => handleCheckout(), 500);
      }
    } catch (e) {
      updateAuthorityProposal(entryId, {
        status: "error",
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  function handleDismissProposal(entryId: string) {
    updateAuthorityProposal(entryId, { status: "dismissed" });
  }

  async function handleBuy(sku: string) {
    if (!sessionId) return;
    setOrderError(null);
    setGuardianResult(null);
    setCheckoutInfo(null);
    setBuyingSku(sku);
    try {
      const newOrder = await selectOrder(sessionId, sku);
      setOrder(newOrder);
    } catch (e) {
      setOrderError(e instanceof Error ? e.message : String(e));
    } finally {
      setBuyingSku(null);
    }
  }

  async function handleApprove() {
    if (!sessionId || !order || orderBusy) return;
    setOrderBusy(true);
    setOrderError(null);
    try {
      const updated = await approveOrder(sessionId, order.id);
      setOrder(updated);
    } catch (e) {
      setOrderError(e instanceof Error ? e.message : String(e));
    } finally {
      setOrderBusy(false);
    }
  }

  async function handleCheckout() {
    if (!sessionId || !order || orderBusy) return;
    setOrderBusy(true);
    setOrderError(null);
    setCheckoutInfo(null);
    const turnId = `checkout-${order.id}-${Date.now()}`;
    startThought(turnId);
    try {
      await streamCheckout(sessionId, order.id, (evt) => {
        if (evt.type === "step") {
          upsertThoughtStep(turnId, evt);
        } else if (evt.type === "result") {
          finishThought(turnId);
          const updatedOrder = evt.order as Order;
          const guardian = evt.guardian as GuardianResult;
          setOrder(updatedOrder);
          setGuardianResult(guardian);
          // V3: the backend only ever includes this when the denial was
          // specifically fixable by granting a fresh delegated authority
          // (see main.py's `_authority_proposal`) - offer the in-chat
          // quick action right here instead of sending the user to the
          // separate panel.
          const proposal = evt.authority_proposal as AuthorityProposal | null | undefined;
          if (proposal) {
            addAuthorityProposal(updatedOrder.id, proposal);
          }
          if (
            evt.payment_mode === "real" &&
            evt.razorpay_order_id &&
            evt.razorpay_key_id &&
            evt.amount_paise
          ) {
            setCheckoutInfo({
              amountPaise: evt.amount_paise as number,
              razorpayOrderId: evt.razorpay_order_id as string,
              razorpayKeyId: evt.razorpay_key_id as string,
            });
          }
        } else if (evt.type === "error") {
          finishThought(turnId);
          setOrderError(evt.message ?? "Something went wrong running the Payment Guardian.");
        }
      });
    } catch (e) {
      finishThought(turnId);
      setOrderError(e instanceof Error ? e.message : String(e));
    } finally {
      setOrderBusy(false);
    }
  }

  function handlePayNow() {
    if (!sessionId || !order || !checkoutInfo || payingNow) return;
    if (typeof window.Razorpay !== "function") {
      setOrderError("Razorpay Checkout script hasn't loaded - check your network connection and reload.");
      return;
    }
    setOrderError(null);
    setPayingNow(true);
    const rzp = new window.Razorpay({
      key: checkoutInfo.razorpayKeyId,
      amount: checkoutInfo.amountPaise,
      currency: order.currency,
      order_id: checkoutInfo.razorpayOrderId,
      name: "Agent Buyer",
      description: order.product_name,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      handler: async (response: any) => {
        try {
          const updated = await verifyPayment(sessionId, order.id, {
            razorpay_order_id: response.razorpay_order_id,
            razorpay_payment_id: response.razorpay_payment_id,
            razorpay_signature: response.razorpay_signature,
          });
          setOrder(updated);
          showPaymentOutcome(
            "success",
            `Payment successful — Order #${updated.id} (${updated.product_name}) paid ${formatPrice(
              updated.unit_price_at_selection * updated.quantity,
              updated.currency,
            )}.`,
          );
        } catch (e) {
          setOrderError(e instanceof Error ? e.message : String(e));
        } finally {
          setPayingNow(false);
        }
      },
      modal: {
        ondismiss: () => setPayingNow(false),
      },
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    rzp.on?.("payment.failed", (response: any) => {
      const err = response?.error ?? {};
      reportPaymentAttemptFailed(sessionId, order.id, {
        razorpay_order_id: err.metadata?.order_id,
        razorpay_payment_id: err.metadata?.payment_id,
        reason: err.reason,
        description: err.description,
      })
        .then((updated) => {
          setOrder(updated);
          showPaymentOutcome(
            "failure",
            `Payment attempt failed for Order #${updated.id} (${updated.product_name})` +
              (err.description ? `: ${err.description}` : ".") +
              " You can retry from the Order panel.",
          );
        })
        .catch((e) => setOrderError(e instanceof Error ? e.message : String(e)))
        .finally(() => setPayingNow(false));
    });
    rzp.open();
  }

  async function handleSimulate(outcome: "success" | "failure") {
    if (!sessionId || !order || orderBusy) return;
    setOrderBusy(true);
    setOrderError(null);
    try {
      const updated = await simulatePayment(sessionId, order.id, outcome);
      setOrder(updated);
      if (outcome === "success") {
        showPaymentOutcome(
          "success",
          `Payment successful (simulated) — Order #${updated.id} (${updated.product_name}) paid ${formatPrice(
            updated.unit_price_at_selection * updated.quantity,
            updated.currency,
          )}.`,
        );
      } else {
        showPaymentOutcome(
          "failure",
          `Payment failed (simulated) for Order #${updated.id} (${updated.product_name}).`,
        );
      }
    } catch (e) {
      setOrderError(e instanceof Error ? e.message : String(e));
    } finally {
      setOrderBusy(false);
    }
  }

  async function handleReset() {
    if (sessionId) await resetSession(sessionId);
    setEntries([]);
    setTurnIntent(EMPTY_CONTEXT);
    setContext(EMPTY_CONTEXT);
    setError(null);
    setRecommendError(null);
    setCandidates([]);
    setShowAllCandidates(false);
    setRecommendedSkus([]);
    setAuditLog([]);
    setAutonomyMode("semi_autonomous");
    setOrder(null);
    setGuardianResult(null);
    setOrderError(null);
    setCheckoutInfo(null);
    setPayingNow(false);
    setAuthority(null);
    setAuthorityError(null);
    clearAutoFlowTimers();
    autoFlowRef.current = { boughtSku: null, checkedOutOrderId: null, paidOrderId: null, deniedOrderId: null };
    lastAutoSearchContextRef.current = null;
    lastAutoBuyContextRef.current = null;
    if (paymentBannerTimerRef.current !== null) {
      window.clearTimeout(paymentBannerTimerRef.current);
      paymentBannerTimerRef.current = null;
    }
    setPaymentBanner(null);
    const id = await newSession();
    setSessionId(id);
  }

  return (
    <div className="layout">
      <div className="chat-pane">
        <div className="header">
          <h1>Agent Buyer</h1>
          <p>
            Tell it what you're shopping for — it searches automatically once it has enough to
            go on, and picks the autonomy mode from how you phrase it (say "just buy the best
            one" for hands-free, or "let me approve first" to stay in control). The Payment
            Guardian decides ALLOW or DENY independently of the LLM either way.
          </p>
          <div className="mode-selector">
            <span className="mode-selector-label" title="Normally set automatically from your message - click to override">
              Mode (auto-detected):
            </span>
            {(Object.keys(MODE_LABELS) as AutonomyMode[]).map((m) => (
              <button
                key={m}
                className={`mode-btn ${autonomyMode === m ? "active" : ""}`}
                onClick={() => handleSetMode(m)}
                title="Manual override - the agent normally sets this itself from your message"
              >
                {MODE_LABELS[m]}
              </button>
            ))}
            {paymentConfig && (
              <span className="gateway-badge" title="Which payment gateway checkout will use">
                {paymentConfig.mode === "real" ? "Razorpay Test Mode" : "Stub gateway (no Razorpay keys yet)"}
              </span>
            )}
          </div>
        </div>
        {paymentBanner && (
          <div className={`payment-banner ${paymentBanner.kind}`} role="status">
            <span>{paymentBanner.kind === "success" ? "✅" : "❌"}</span>
            <span>{paymentBanner.text}</span>
            <button className="payment-banner-dismiss" onClick={() => setPaymentBanner(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}
        <div className="messages" ref={scrollRef}>
          {entries.map((e, i) => {
            if (e.kind === "message") {
              return (
                <div key={i} className={`bubble ${e.role}`}>
                  {e.content}
                </div>
              );
            }
            if (e.kind === "thought") {
              return <ThoughtStream key={e.turnId} steps={e.steps} active={e.active} />;
            }
            // V3: in-chat "Grant authority" quick action - Claude-style
            // permission prompt. The agent proposes the values; only the
            // [Grant] click actually authorizes anything.
            const { proposal } = e;
            return (
              <div key={e.id} className="bubble assistant authority-proposal">
                <div>
                  Fully-autonomous purchasing needs delegated authority first
                  {proposal.reason ? ` (${proposal.reason})` : ""}. Grant{" "}
                  <strong>{formatPrice(proposal.suggested_max_spend, proposal.currency)}</strong> for{" "}
                  <strong>{proposal.category === "any" ? "any category" : proposal.category}</strong>,
                  valid <strong>{proposal.duration_minutes} min</strong>?
                </div>
                {e.status === "pending" && (
                  <div className="authority-proposal-actions">
                    <button onClick={() => handleGrantFromProposal(e.id, e.orderId, proposal)}>
                      Grant
                    </button>
                    <button className="secondary" onClick={() => handleDismissProposal(e.id)}>
                      Not now
                    </button>
                  </div>
                )}
                {e.status === "granting" && <div className="authority-proposal-status">Granting…</div>}
                {e.status === "granted" && (
                  <div className="authority-proposal-status">✓ Granted — retrying purchase.</div>
                )}
                {e.status === "dismissed" && (
                  <div className="authority-proposal-status">
                    Not granted. You can still grant authority from the panel on the right.
                  </div>
                )}
                {e.status === "error" && (
                  <div className="authority-proposal-status error">{e.error}</div>
                )}
              </div>
            );
          })}
          {error && <div className="bubble error">{error}</div>}
          {recommendError && <div className="bubble error">{recommendError}</div>}
        </div>
        <div className="composer">
          <input
            value={input}
            placeholder="Find me good running shoes under ₹4,000 for marathon training"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={sending || !sessionId}
          />
          <button onClick={handleSend} disabled={sending || !sessionId}>
            {sending ? "…" : "Send"}
          </button>
          <button
            className="recommend-btn"
            onClick={handleRecommend}
            disabled={!canRecommend || recommending || !sessionId}
            title={
              canRecommend
                ? "Search now without waiting for the agent to decide it's ready (optional - it searches automatically once it has enough context)"
                : "Tell it what you want first"
            }
          >
            {recommending ? "Searching…" : "Search now"}
          </button>
        </div>
      </div>
      <div className="context-pane">
        <button className="reset-btn" onClick={handleReset}>
          Reset conversation
        </button>
        <div style={{ height: 12 }} />

        {autonomyMode === "fully_autonomous" && (
          <div className="context-card">
            <h2>Delegated Authority</h2>
            <p className="sub">
              Required before Fully-Autonomous purchases can be ALLOWed — the Guardian checks
              every order against this max spend, category, and expiry.
            </p>
            {authority && authority.active ? (
              <>
                <div className="context-row">
                  <span className="k">Remaining</span>
                  <span className="v">
                    {formatPrice(authority.remaining, authority.currency)} of{" "}
                    {formatPrice(authority.max_spend, authority.currency)}
                  </span>
                </div>
                <div className="context-row">
                  <span className="k">Category</span>
                  <span className="v">
                    {authority.category === "any" ? "Any category" : authority.category}
                  </span>
                </div>
                <div className="context-row">
                  <span className="k">Expires</span>
                  <span className="v">{new Date(authority.expires_at).toLocaleString()}</span>
                </div>
                <button
                  className="revoke-authority-btn"
                  onClick={handleRevokeAuthority}
                  disabled={authorityBusy}
                >
                  {authorityBusy ? "…" : "Revoke authority"}
                </button>
              </>
            ) : (
              <div className="authority-form">
                {authority && !authority.active && (
                  <p className="authority-expired">
                    {authority.revoked
                      ? "The previous authority was revoked."
                      : "The previous authority expired."}{" "}
                    Grant a new one to continue.
                  </p>
                )}
                <label className="authority-field">
                  <span>Max spend (INR)</span>
                  <input
                    type="number"
                    min="1"
                    value={grantMaxSpend}
                    onChange={(e) => setGrantMaxSpend(e.target.value)}
                  />
                </label>
                <label className="authority-field">
                  <span>Category</span>
                  <select value={grantCategory} onChange={(e) => setGrantCategory(e.target.value)}>
                    <option value="any">Any category</option>
                    {categories.map((cat) => (
                      <option key={cat} value={cat}>
                        {cat}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="authority-field">
                  <span>Valid for (minutes)</span>
                  <input
                    type="number"
                    min="1"
                    value={grantDurationMinutes}
                    onChange={(e) => setGrantDurationMinutes(e.target.value)}
                  />
                </label>
                <button className="grant-authority-btn" onClick={handleGrantAuthority} disabled={authorityBusy}>
                  {authorityBusy ? "…" : "Grant authority"}
                </button>
              </div>
            )}
            {authorityError && <p className="order-error">{authorityError}</p>}
          </div>
        )}

        <ContextPanel
          title="This turn"
          subtitle="What the LLM extracted from your last message alone"
          context={turnIntent}
        />
        <ContextPanel
          title="Accumulated context"
          subtitle="Merged across the whole conversation — this drives 'Find products'"
          context={context}
        />

        {auditLog.length > 0 && (
          <Collapsible
            title="Activity log"
            summary={`${auditLog.length} step${auditLog.length === 1 ? "" : "s"}`}
          >
            <p className="sub">What the agent actually did for this recommendation</p>
            <AuditLog entries={auditLog} />
          </Collapsible>
        )}

        {candidates.length > 0 && (
          <div className="context-card">
            <h2>Candidates</h2>
            <p className="sub">
              {candidates.length} product(s) retrieved from the catalog — highlighted ones are the agent's picks.
              Current mode: <strong>{MODE_LABELS[autonomyMode]}</strong>.
            </p>
            {autonomyMode === "fully_autonomous" && !order && (
              <p className="autonomous-hint">
                Delegated authority is in place — the agent will auto-select its top pick, run the
                Payment Guardian, and complete payment without waiting for a click.
              </p>
            )}
            <div className="product-grid">
              {(showAllCandidates ? candidates : candidates.slice(0, CANDIDATE_PREVIEW_COUNT)).map((p) => (
                <ProductCard
                  key={p.id}
                  product={p}
                  recommended={recommendedSkus.includes(p.sku)}
                  onBuy={handleBuy}
                  buying={buyingSku === p.sku}
                />
              ))}
            </div>
            {candidates.length > CANDIDATE_PREVIEW_COUNT && (
              <button
                type="button"
                className="show-more-btn"
                onClick={() => setShowAllCandidates((s) => !s)}
              >
                {showAllCandidates
                  ? "Show fewer"
                  : `Show ${candidates.length - CANDIDATE_PREVIEW_COUNT} more`}
              </button>
            )}
          </div>
        )}

        {orderError && (
          <div className="context-card">
            <p className="order-error">{orderError}</p>
          </div>
        )}

        {order && (
          <div className="context-card">
            <h2>Order</h2>
            <p className="sub">
              {/* Bugfix round 15: `?? order.autonomy_mode` fallback - an
                  order created before Recommendation mode was removed can
                  still have that literal string stored in the DB, which
                  is no longer a key in MODE_LABELS and would otherwise
                  render as nothing. */}
              #{order.id} · {order.merchant} · mode: {MODE_LABELS[order.autonomy_mode] ?? order.autonomy_mode}
            </p>
            <div className="order-summary">
              <div>
                <div className="product-name">{order.product_name}</div>
                <div className="product-brand">
                  {formatPrice(order.unit_price_at_selection, order.currency)} × {order.quantity}
                </div>
              </div>
              <span className={`status-badge ${statusClass(order.status)}`}>{order.status}</span>
            </div>

            <div className="order-actions">
              {order.autonomy_mode === "fully_autonomous" &&
                ["SELECTED", "PAYMENT_PENDING"].includes(order.status) && (
                  <p className="autonomous-hint">
                    {order.status === "SELECTED"
                      ? "Running the Payment Guardian automatically…"
                      : "Delegated authority is active — completing payment automatically…"}
                  </p>
                )}
              {order.status === "SELECTED" &&
                order.autonomy_mode === "semi_autonomous" &&
                !order.user_approved && (
                  <button className="approve-btn" onClick={handleApprove} disabled={orderBusy}>
                    {orderBusy ? "…" : "Approve purchase"}
                  </button>
                )}
              {order.status === "SELECTED" && order.autonomy_mode !== "fully_autonomous" && (
                <button
                  className="checkout-btn"
                  onClick={handleCheckout}
                  disabled={
                    orderBusy ||
                    (order.autonomy_mode === "semi_autonomous" && !order.user_approved)
                  }
                  title={
                    order.autonomy_mode === "semi_autonomous" && !order.user_approved
                      ? "Approve the order first"
                      : "Run the Payment Guardian and proceed if allowed"
                  }
                >
                  {orderBusy ? "…" : "Checkout (run Guardian)"}
                </button>
              )}
              {order.status === "PAYMENT_PENDING" &&
                paymentConfig?.mode === "stub" &&
                order.autonomy_mode !== "fully_autonomous" && (
                  <>
                    <button className="sim-success-btn" onClick={() => handleSimulate("success")} disabled={orderBusy}>
                      Simulate payment success
                    </button>
                    <button className="sim-fail-btn" onClick={() => handleSimulate("failure")} disabled={orderBusy}>
                      Simulate payment failure
                    </button>
                  </>
                )}
              {order.status === "PAYMENT_PENDING" &&
                paymentConfig?.mode === "real" &&
                checkoutInfo &&
                order.autonomy_mode !== "fully_autonomous" && (
                  <button className="pay-btn" onClick={handlePayNow} disabled={payingNow}>
                    {payingNow ? "Waiting for payment…" : "Pay now (Razorpay)"}
                  </button>
                )}
              {order.status === "FAILED" && (
                <button className="buy-btn" onClick={() => handleBuy(order.sku)} disabled={orderBusy}>
                  Re-select and try again
                </button>
              )}
            </div>

            {guardianResult && <GuardianPanel result={guardianResult} />}

            <div style={{ marginTop: 12 }}>
              <Collapsible
                title="Order history"
                summary={`${order.events.length} event${order.events.length === 1 ? "" : "s"}`}
                variant="inline"
              >
                <OrderEventLog order={order} />
              </Collapsible>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
