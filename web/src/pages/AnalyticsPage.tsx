import { useEffect, useState } from "react";
import type { QueryRecord, StatsData } from "../api/types";
import { api } from "../api/client";
import { Skeleton } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";
import { timeAgo } from "../utils/time";

const PAGE_SIZE = 20;

export default function AnalyticsPage() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Analytics</h1>
      <StatsOverview />
      <QueryHistory />
    </div>
  );
}

// --- Stats Overview ---

function StatsOverview() {
  const [stats, setStats] = useState<StatsData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.stats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
        <Skeleton className="h-24 rounded-xl" />
      </div>
    );
  }

  if (!stats) return null;

  const topInterface = Object.entries(stats.queries_by_interface).sort((a, b) => b[1] - a[1])[0];
  const topType = Object.entries(stats.queries_by_type).sort((a, b) => b[1] - a[1])[0];

  return (
    <div className="mb-10">
      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Queries" value={stats.total_queries} />
        <StatCard label="Avg Latency" value={`${Math.round(stats.avg_latency_ms)}ms`} />
        <StatCard label="Top Interface" value={topInterface ? topInterface[0] : "--"} />
        <StatCard label="Top Query Type" value={topType ? topType[0] : "--"} />
      </div>

      {/* Latency percentile cards */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard label="p50 Latency" value={`${Math.round(stats.latency_p50)}ms`} />
        <StatCard label="p95 Latency" value={`${Math.round(stats.latency_p95)}ms`} />
        <StatCard label="p99 Latency" value={`${Math.round(stats.latency_p99)}ms`} />
      </div>

      {/* Bar charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <BarChart title="Queries by Interface" data={stats.queries_by_interface} />
        <BarChart title="Queries by Type" data={stats.queries_by_type} />
      </div>

      {/* Top collections */}
      {stats.top_collections.length > 0 && (
        <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
          <h3 className="text-sm font-bold mb-4" style={{ color: "var(--color-text-muted)" }}>Top Collections</h3>
          <div className="space-y-2">
            {stats.top_collections.map((c, i) => (
              <div key={c.name} className="flex items-center gap-3 text-sm">
                <span className="w-5 text-right" style={{ color: "var(--color-text-muted)" }}>{i + 1}.</span>
                <span className="flex-1 font-bold">{c.name}</span>
                <span style={{ color: "var(--color-text-muted)" }}>{c.count} queries</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl p-5" style={{ background: "var(--color-surface)" }}>
      <p className="text-2xl font-bold mb-1">{value}</p>
      <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>{label}</p>
    </div>
  );
}

function BarChart({ title, data }: { title: string; data: Record<string, number> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1]);
  const max = entries[0]?.[1] ?? 1;

  if (entries.length === 0) return null;

  return (
    <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
      <h3 className="text-sm font-bold mb-4" style={{ color: "var(--color-text-muted)" }}>{title}</h3>
      <div className="space-y-3">
        {entries.map(([label, count]) => (
          <div key={label}>
            <div className="flex justify-between text-sm mb-1">
              <span>{label}</span>
              <span style={{ color: "var(--color-text-muted)" }}>{count}</span>
            </div>
            <div className="h-2 rounded-full" style={{ background: "var(--color-surface-2)" }}>
              <div
                className="h-2 rounded-full transition-all duration-300"
                style={{
                  width: `${Math.max(4, (count / max) * 100)}%`,
                  background: "var(--color-accent)",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Query History ---

function QueryHistory() {
  const [queries, setQueries] = useState<QueryRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [iface, setIface] = useState("");
  const [queryType, setQueryType] = useState("");

  const fetchQueries = (off: number, ifa: string, qt: string) => {
    setLoading(true);
    setError(null);
    api.queries({
      limit: PAGE_SIZE,
      offset: off,
      interface: ifa || undefined,
      query_type: qt || undefined,
    })
      .then((data) => {
        setQueries(data.queries);
        setTotal(data.total);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to fetch queries"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchQueries(offset, iface, queryType);
  }, [offset, iface, queryType]);

  const showingFrom = total > 0 ? offset + 1 : 0;
  const showingTo = Math.min(offset + PAGE_SIZE, total);

  return (
    <div>
      <h2 className="text-lg font-bold mb-4">Query History</h2>

      {/* Filters */}
      <div className="flex gap-3 mb-6">
        <select
          value={iface}
          onChange={(e) => { setIface(e.target.value); setOffset(0); }}
          className="h-10 px-3 rounded-lg text-sm outline-none cursor-pointer"
          style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
        >
          <option value="">All interfaces</option>
          <option value="api">API</option>
          <option value="cli">CLI</option>
          <option value="mcp">MCP</option>
          <option value="web">Web</option>
        </select>
        <select
          value={queryType}
          onChange={(e) => { setQueryType(e.target.value); setOffset(0); }}
          className="h-10 px-3 rounded-lg text-sm outline-none cursor-pointer"
          style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
        >
          <option value="">All types</option>
          <option value="search">Search</option>
          <option value="qa">Q&A</option>
        </select>
      </div>

      {error && (
        <div className="rounded-lg px-4 py-3 text-sm mb-6" style={{ background: "var(--color-surface)", color: "var(--color-error)" }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 w-full rounded-lg" />
          ))}
        </div>
      ) : queries.length === 0 ? (
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-12 h-12">
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
            </svg>
          }
          title="No queries yet"
          description="Run some searches or ask questions to see query history."
        />
      ) : (
        <>
          <div className="overflow-x-auto rounded-xl" style={{ background: "var(--color-surface)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Query</th>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Type</th>
                  <th className="text-left px-4 py-3 font-bold hidden md:table-cell" style={{ color: "var(--color-text-muted)" }}>Mode</th>
                  <th className="text-right px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Results</th>
                  <th className="text-right px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Latency</th>
                  <th className="text-left px-4 py-3 font-bold hidden lg:table-cell" style={{ color: "var(--color-text-muted)" }}>Interface</th>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Time</th>
                </tr>
              </thead>
              <tbody>
                {queries.map((q) => (
                  <tr key={q.id} style={{ borderBottom: "1px solid var(--color-border)" }}>
                    <td className="px-4 py-3 max-w-xs truncate">{q.query_text}</td>
                    <td className="px-4 py-3">
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{ background: "var(--color-surface-2)", color: "var(--color-text-muted)" }}
                      >
                        {q.query_type}
                      </span>
                    </td>
                    <td className="px-4 py-3 hidden md:table-cell" style={{ color: "var(--color-text-muted)" }}>{q.search_mode}</td>
                    <td className="px-4 py-3 text-right">{q.result_count}</td>
                    <td className="px-4 py-3 text-right" style={{ color: "var(--color-text-muted)" }}>{Math.round(q.latency_ms)}ms</td>
                    <td className="px-4 py-3 hidden lg:table-cell" style={{ color: "var(--color-text-muted)" }}>{q.interface}</td>
                    <td className="px-4 py-3 text-xs" style={{ color: "var(--color-text-muted)" }}>{timeAgo(q.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between mt-4 text-sm" style={{ color: "var(--color-text-muted)" }}>
            <span>Showing {showingFrom}–{showingTo} of {total}</span>
            <div className="flex gap-2">
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                className="px-3 py-1.5 rounded-lg text-sm cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-150"
                style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
              >
                Prev
              </button>
              <button
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                className="px-3 py-1.5 rounded-lg text-sm cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed transition-colors duration-150"
                style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
