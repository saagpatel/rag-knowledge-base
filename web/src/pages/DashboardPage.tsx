import { useEffect, useState } from "react";
import type { CollectionInfo, DocumentListData, HealthData, QueryRecord, StatsData } from "../api/types";
import { api } from "../api/client";
import { SkeletonCard } from "../components/Skeleton";
import StatusBadge from "../components/StatusBadge";
import { timeAgo } from "../utils/time";

interface DashboardData {
  health: HealthData | null;
  collections: CollectionInfo[] | null;
  documents: DocumentListData | null;
  stats: StatsData | null;
  recentQueries: QueryRecord[] | null;
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData>({
    health: null,
    collections: null,
    documents: null,
    stats: null,
    recentQueries: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    Promise.allSettled([
      api.health(),
      api.collections.list(),
      api.documents.list({ limit: 1 }),
      api.stats(),
      api.queries({ limit: 5 }),
    ]).then(([healthR, collectionsR, documentsR, statsR, queriesR]) => {
      if (cancelled) return;

      const health = healthR.status === "fulfilled" ? healthR.value : null;
      if (!health) {
        setError(healthR.status === "rejected" ? String(healthR.reason) : "Failed to fetch health");
      }

      setData({
        health,
        collections: collectionsR.status === "fulfilled" ? collectionsR.value : null,
        documents: documentsR.status === "fulfilled" ? documentsR.value : null,
        stats: statsR.status === "fulfilled" ? statsR.value : null,
        recentQueries: queriesR.status === "fulfilled" ? queriesR.value.queries : null,
      });
      setLoading(false);
    });

    return () => { cancelled = true; };
  }, []);

  if (error && !data.health) {
    return (
      <div>
        <h1 className="text-2xl font-bold mb-8">Dashboard</h1>
        <div className="rounded-xl p-6" style={{ background: "var(--color-surface)", border: "1px solid var(--color-error)" }}>
          <p style={{ color: "var(--color-error)" }}>Failed to connect: {error}</p>
          <p className="text-sm mt-2" style={{ color: "var(--color-text-muted)" }}>
            Make sure the API server is running on port 8000.
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div>
        <h1 className="text-2xl font-bold mb-8">Dashboard</h1>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  const { health, collections, documents, stats, recentQueries } = data;

  return (
    <div>
      <div className="flex items-center gap-4 mb-10">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {health && <StatusBadge status={health.status} />}
      </div>

      {/* Row 1: Service health */}
      {health && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card title="Ollama">
            <StatusBadge status={health.ollama.status} label={health.ollama.status === "ok" ? "Connected" : "Disconnected"} />
            {health.ollama.detail && (
              <p className="text-xs mt-2" style={{ color: "var(--color-text-muted)" }}>{health.ollama.detail}</p>
            )}
          </Card>

          <Card title="Qdrant">
            <StatusBadge status={health.qdrant.status} label={health.qdrant.status === "ok" ? "Connected" : "Disconnected"} />
            {health.qdrant.detail && (
              <p className="text-xs mt-2" style={{ color: "var(--color-text-muted)" }}>{health.qdrant.detail}</p>
            )}
          </Card>

          <Card title="System">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span style={{ color: "var(--color-text-muted)" }}>Version</span>
                <span className="font-bold">{health.version}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: "var(--color-text-muted)" }}>Uptime</span>
                <span className="font-bold">{formatUptime(health.uptime_seconds)}</span>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Row 2: Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Documents" value={documents?.total ?? "--"} />
        <StatCard label="Collections" value={collections?.length ?? "--"} />
        <StatCard label="Total Queries" value={stats?.total_queries ?? "--"} />
        <StatCard label="Avg Latency" value={stats ? `${Math.round(stats.avg_latency_ms)}ms` : "--"} />
      </div>

      {/* Row 3: Recent queries */}
      {recentQueries && recentQueries.length > 0 && (
        <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
          <h2 className="text-sm font-bold mb-4" style={{ color: "var(--color-text-muted)" }}>Recent Queries</h2>
          <div className="space-y-3">
            {recentQueries.map((q) => (
              <div key={q.id} className="flex items-center gap-3 text-sm">
                <span className="flex-1 truncate">{q.query_text}</span>
                <span
                  className="text-xs px-2 py-0.5 rounded-full shrink-0"
                  style={{ background: "var(--color-surface-2)", color: "var(--color-text-muted)" }}
                >
                  {q.query_type}
                </span>
                <span className="text-xs shrink-0" style={{ color: "var(--color-text-muted)" }}>
                  {Math.round(q.latency_ms)}ms
                </span>
                <span className="text-xs shrink-0" style={{ color: "var(--color-text-muted)" }}>
                  {timeAgo(q.created_at)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
      <h2 className="text-sm font-bold mb-4" style={{ color: "var(--color-text-muted)" }}>
        {title}
      </h2>
      {children}
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

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}
