import { useState } from "react";
import type { ThoughtStep } from "./types";

/**
 * V2 "thought stream" - the agent's live, animated reasoning for a single
 * turn, modeled on how Claude shows thinking: collapsed by default, a
 * single line that updates in place as the current step changes (spinner +
 * that step's own label - not a growing list left open on screen), and a
 * click to expand into the full, detailed step-by-step list (each step
 * itself further click-to-expand into its raw detail: the actual
 * accumulated-context JSON, a Guardian check's full explanation, a raw
 * search result count).
 *
 * One continuous stream per turn: context extraction, product search, and
 * every Guardian check all append to the SAME ordered list under the SAME
 * header, in the order they actually happened on the backend - not three
 * separate collapsible panels. `active` means the SSE connection for this
 * turn is still open; once it closes the header switches from the live
 * current-step line to a plain "Thought process" summary, matching
 * Claude's own collapsed thinking block once a turn is done. Collapsed is
 * the default in both states - expanding is an explicit user choice, kept
 * only until they collapse it again.
 */
export function ThoughtStream({
  steps,
  active,
  startCollapsed = true,
}: {
  steps: ThoughtStep[];
  active: boolean;
  startCollapsed?: boolean;
}) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [collapsed, setCollapsed] = useState(startCollapsed);

  if (steps.length === 0 && !active) return null;

  function toggleStep(id: string) {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const doneCount = steps.filter((s) => s.status !== "pending").length;
  const errorCount = steps.filter((s) => s.status === "error").length;
  const currentStep = steps[steps.length - 1];

  return (
    <div className={`thought-stream ${active ? "active" : "settled"}`}>
      <button
        type="button"
        className="thought-stream-header"
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        {active ? (
          <span className="thought-spinner" aria-hidden="true" />
        ) : (
          <span className={`thought-done-dot ${errorCount > 0 ? "error" : ""}`} aria-hidden="true" />
        )}
        <span className="thought-stream-title">
          {collapsed && active && currentStep ? currentStep.label : active ? "Thinking…" : "Thought process"}
        </span>
        <span className="thought-stream-count">
          {doneCount} step{doneCount === 1 ? "" : "s"}
        </span>
        <span className="thought-stream-caret">{collapsed ? "▸" : "▾"}</span>
      </button>
      {!collapsed && (
        <ul className="thought-stream-list">
          {steps.map((s) => (
            <li key={s.id} className={`thought-step thought-step--${s.status}`}>
              <button
                type="button"
                className="thought-step-row"
                onClick={() => s.detail && toggleStep(s.id)}
                disabled={!s.detail}
              >
                <span className={`thought-step-dot thought-step-dot--${s.status}`} aria-hidden="true" />
                <span className="thought-step-label">{s.label}</span>
                {s.detail && (
                  <span className="thought-step-caret">{expandedIds.has(s.id) ? "▾" : "▸"}</span>
                )}
              </button>
              {s.detail && expandedIds.has(s.id) && (
                <pre className="thought-step-detail">{s.detail}</pre>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
