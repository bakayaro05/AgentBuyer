import type {
  AutonomyMode,
  ChatResponse,
  CheckoutResponse,
  DelegatedAuthority,
  Order,
  PaymentConfig,
  RecommendResponse,
  StreamEvent,
  VerifyPaymentPayload,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

// Generic SSE-over-POST consumer for the V2 "thought stream" endpoints
// (/api/chat/stream, /api/recommend/stream, /api/orders/{id}/checkout/stream).
// EventSource can't POST a body, so this reads the fetch response body as
// a stream directly and splits it on the SSE record separator ("\n\n")
// itself - each backend frame is a single `data: <json>` line. Resolves
// once the stream ends; callers get progress purely through `onEvent`.
async function streamPost(
  path: string,
  body: unknown,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let message = text;
    try {
      message = JSON.parse(text)?.detail ?? text;
    } catch {
      // not JSON - use the raw text
    }
    throw new Error(message || `Request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) >= 0) {
      const rawRecord = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);
      const dataLine = rawRecord.split("\n").find((line) => line.startsWith("data:"));
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine.slice(5).trim()) as StreamEvent);
      } catch {
        // malformed/partial chunk - skip rather than crash the stream
      }
    }
  }
}

export async function streamChat(
  sessionId: string,
  message: string,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  return streamPost("/api/chat/stream", { session_id: sessionId, message }, onEvent);
}

export async function streamRecommend(
  sessionId: string,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  return streamPost("/api/recommend/stream", { session_id: sessionId }, onEvent);
}

export async function streamCheckout(
  sessionId: string,
  orderId: number,
  onEvent: (evt: StreamEvent) => void,
): Promise<void> {
  return streamPost(`/api/orders/${orderId}/checkout/stream`, { session_id: sessionId }, onEvent);
}

export async function newSession(): Promise<string> {
  const res = await fetch(`${API_BASE}/api/session/new`, { method: "POST" });
  const data = await handle<{ session_id: string }>(res);
  return data.session_id;
}

export async function sendChat(
  sessionId: string,
  message: string,
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  return handle<ChatResponse>(res);
}

export async function resetSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function recommend(sessionId: string): Promise<RecommendResponse> {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return handle<RecommendResponse>(res);
}

export async function setMode(sessionId: string, mode: AutonomyMode): Promise<void> {
  await fetch(`${API_BASE}/api/session/mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, mode }),
  });
}

export async function selectOrder(
  sessionId: string,
  sku: string,
  quantity = 1,
): Promise<Order> {
  const res = await fetch(`${API_BASE}/api/orders/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, sku, quantity }),
  });
  return handle<Order>(res);
}

export async function approveOrder(sessionId: string, orderId: number): Promise<Order> {
  const res = await fetch(`${API_BASE}/api/orders/${orderId}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return handle<Order>(res);
}

export async function checkoutOrder(
  sessionId: string,
  orderId: number,
): Promise<CheckoutResponse> {
  const res = await fetch(`${API_BASE}/api/orders/${orderId}/checkout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return handle<CheckoutResponse>(res);
}

export async function simulatePayment(
  sessionId: string,
  orderId: number,
  outcome: "success" | "failure",
): Promise<Order> {
  const res = await fetch(`${API_BASE}/api/orders/${orderId}/simulate-payment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, outcome }),
  });
  return handle<Order>(res);
}

export async function getPaymentConfig(): Promise<PaymentConfig> {
  const res = await fetch(`${API_BASE}/api/payment-config`);
  return handle<PaymentConfig>(res);
}

export async function verifyPayment(
  sessionId: string,
  orderId: number,
  payload: VerifyPaymentPayload,
): Promise<Order> {
  const res = await fetch(`${API_BASE}/api/orders/${orderId}/verify-payment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...payload }),
  });
  return handle<Order>(res);
}

export async function reportPaymentAttemptFailed(
  sessionId: string,
  orderId: number,
  payload: {
    razorpay_order_id?: string;
    razorpay_payment_id?: string;
    reason?: string;
    description?: string;
  },
): Promise<Order> {
  const res = await fetch(`${API_BASE}/api/orders/${orderId}/payment-attempt-failed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...payload }),
  });
  return handle<Order>(res);
}

export async function getCategories(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/categories`);
  const data = await handle<{ categories: string[] }>(res);
  return data.categories;
}

export async function getAuthority(sessionId: string): Promise<DelegatedAuthority | null> {
  const res = await fetch(`${API_BASE}/api/authority?session_id=${encodeURIComponent(sessionId)}`);
  return handle<DelegatedAuthority | null>(res);
}

export async function grantAuthority(
  sessionId: string,
  payload: { max_spend: number; category: string; duration_minutes: number },
): Promise<DelegatedAuthority> {
  const res = await fetch(`${API_BASE}/api/authority/grant`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, ...payload }),
  });
  return handle<DelegatedAuthority>(res);
}

export async function revokeAuthority(sessionId: string): Promise<DelegatedAuthority> {
  const res = await fetch(`${API_BASE}/api/authority/revoke`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return handle<DelegatedAuthority>(res);
}
