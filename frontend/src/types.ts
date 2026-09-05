export interface ShoppingContext {
  category: string | null;
  keywords: string[];
  min_price: number | null;
  max_price: number | null;
  currency: string;
  attributes: Record<string, string>;
  notes: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  turn_intent: ShoppingContext;
  context: ShoppingContext;
  history: ChatMessage[];
  ready_to_search?: boolean;
  desired_mode?: string | null;
  mode_changed?: string | null;
  // Bugfix round 11: raw regex-based signal that this message itself is a
  // bare purchase confirmation ("buy it") - see schemas.py's ChatResponse
  // for the full rationale. Consumed by App.tsx's handleSend to decide
  // whether to auto-select an already-recommended candidate.
  buy_confirmation?: boolean;
}

export interface Product {
  id: number;
  sku: string;
  merchant: string | null;
  name: string;
  brand: string;
  category: string;
  description: string;
  price: number;
  currency: string;
  rating: number;
  review_count: number;
  in_stock: boolean;
  stock_qty: number;
  attributes: Record<string, string>;
}

export interface AuditEntry {
  step: string;
  detail: string;
}

export interface RecommendResponse {
  reply: string;
  candidates: Product[];
  recommended_skus: string[];
  audit_log: AuditEntry[];
}

// Bugfix round 15: "recommendation" removed as an autonomy mode -
// Semi-Autonomous already did everything it did (search, reason, show
// candidates) plus buy-on-approval, so it was a strict subset with no
// purpose of its own. Semi-Autonomous is now the default/floor mode.
export type AutonomyMode = "semi_autonomous" | "fully_autonomous";

export interface GuardianCheck {
  name: string;
  passed: boolean;
  detail: string;
}

export interface GuardianResult {
  allowed: boolean;
  reason: string;
  checks: GuardianCheck[];
}

export interface OrderEvent {
  step: string;
  detail: string;
  at: string;
}

export interface Order {
  id: number;
  sku: string;
  product_name: string;
  merchant: string;
  unit_price_at_selection: number;
  currency: string;
  quantity: number;
  autonomy_mode: AutonomyMode;
  status: string;
  user_approved: boolean;
  guardian_allowed: boolean | null;
  guardian_reason: string | null;
  razorpay_order_id: string | null;
  razorpay_payment_id: string | null;
  events: OrderEvent[];
  created_at: string;
  updated_at: string;
}

// V3: when a fully-autonomous checkout is denied specifically for a
// delegated-authority problem (none granted / revoked / expired / wrong
// category / insufficient budget), the backend proposes concrete grant
// values - computed server-side from the order's own price/category,
// never by the LLM - so the chat can offer an in-chat "Grant authority"
// quick action (Claude's own permission-prompt pattern: propose, then a
// human clicks to actually authorize) instead of sending the user to the
// separate Delegated Authority panel.
export interface AuthorityProposal {
  suggested_max_spend: number;
  currency: string;
  category: string;
  duration_minutes: number;
  reason: string;
}

export interface CheckoutResponse {
  order: Order;
  guardian: GuardianResult;
  payment_mode: "stub" | "real" | null;
  razorpay_order_id: string | null;
  razorpay_key_id: string | null;
  amount_paise: number | null;
  authority_proposal: AuthorityProposal | null;
}

export interface PaymentConfig {
  mode: "stub" | "real";
  key_id: string | null;
}

export interface VerifyPaymentPayload {
  razorpay_order_id: string;
  razorpay_payment_id: string;
  razorpay_signature: string;
}

export interface DelegatedAuthority {
  id: number;
  max_spend: number;
  spent: number;
  remaining: number;
  currency: string;
  category: string;
  revoked: boolean;
  expired: boolean;
  active: boolean;
  created_at: string;
  expires_at: string;
}

// V2 "thought stream" - one live, animated step in the agent's visible
// reasoning for a single turn (reading a message, extracting context,
// searching, a Guardian check, ...). `status` drives the dot/spinner in
// the UI; `detail`, when present, is the full data behind the one-liner
// (raw context JSON, a Guardian check's full explanation) and is what
// the click-to-expand affordance reveals.
export interface ThoughtStep {
  id: string;
  label: string;
  detail?: string;
  status: "pending" | "done" | "error";
}

// One SSE frame from a /stream endpoint. `type: "step"` upserts a
// ThoughtStep by id; `type: "result"` carries the same payload shape the
// non-streaming endpoint would have returned (fields vary by endpoint,
// so this is intentionally loose); `type: "error"` mirrors an HTTPException.
export interface StreamEvent {
  type: "step" | "result" | "error";
  id?: string;
  label?: string;
  detail?: string;
  status?: "pending" | "done" | "error";
  message?: string;
  [key: string]: unknown;
}
