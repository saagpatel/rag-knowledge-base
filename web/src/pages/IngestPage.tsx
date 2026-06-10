import { useEffect, useState } from "react";
import type { CollectionInfo, IngestData } from "../api/types";
import { api } from "../api/client";

export default function IngestPage() {
  const [path, setPath] = useState("");
  const [collection, setCollection] = useState("default");
  const [chunkSize, setChunkSize] = useState(512);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [patterns, setPatterns] = useState("");
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [result, setResult] = useState<IngestData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.collections.list().then(setCollections).catch(() => {});
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!path.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const patternList = patterns.trim()
      ? patterns.split(",").map((p) => p.trim()).filter(Boolean)
      : undefined;

    api.ingest({
      path: path.trim(),
      collection,
      chunk_size: chunkSize,
      chunk_overlap: chunkOverlap,
      patterns: patternList,
    })
      .then(setResult)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Ingestion failed"))
      .finally(() => setLoading(false));
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Ingest Documents</h1>

      <form onSubmit={handleSubmit} className="mb-8 space-y-4">
        <input
          type="text"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="File or directory path (e.g. /Users/you/docs)"
          className="w-full h-12 px-5 rounded-lg text-sm outline-none focus:ring-2"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text)",
            fontFamily: "var(--font-mono)",
          }}
        />

        <div className="flex flex-wrap items-center gap-4">
          <select
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            className="h-10 px-3 rounded-lg text-sm outline-none cursor-pointer"
            style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
          >
            {collections.length === 0 && <option value="default">default</option>}
            {collections.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>

          <label className="flex items-center gap-2 text-sm" style={{ color: "var(--color-text-muted)" }}>
            Chunk size
            <input
              type="number"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              min={64}
              max={4096}
              className="w-20 h-10 px-2 rounded-lg text-sm text-center outline-none"
              style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm" style={{ color: "var(--color-text-muted)" }}>
            Overlap
            <input
              type="number"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Number(e.target.value))}
              min={0}
              max={512}
              className="w-20 h-10 px-2 rounded-lg text-sm text-center outline-none"
              style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
            />
          </label>

          <input
            type="text"
            value={patterns}
            onChange={(e) => setPatterns(e.target.value)}
            placeholder="Patterns (e.g. *.md, *.py)"
            className="h-10 px-3 rounded-lg text-sm outline-none flex-1 min-w-40"
            style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
          />

          <button
            type="submit"
            disabled={loading || !path.trim()}
            className="px-6 h-10 rounded-lg text-sm font-bold text-white transition-colors duration-150 cursor-pointer disabled:opacity-50"
            style={{ background: "var(--color-accent)" }}
          >
            {loading ? "Ingesting..." : "Ingest"}
          </button>
        </div>
      </form>

      {error && (
        <div className="rounded-lg px-4 py-3 text-sm mb-6" style={{ background: "var(--color-surface)", color: "var(--color-error)" }}>
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total" value={result.total_files} />
            <StatCard label="Processed" value={result.processed} color="var(--color-success)" />
            <StatCard label="Failed" value={result.failed} color={result.failed > 0 ? "var(--color-error)" : undefined} />
            <StatCard label="Skipped" value={result.skipped} />
          </div>

          {result.results.length > 0 && (
            <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
              <table className="w-full text-sm">
                <thead>
                  <tr style={{ background: "var(--color-surface)" }}>
                    <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>File</th>
                    <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Status</th>
                    <th className="text-right px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Chunks</th>
                    <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Error</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.map((r, i) => (
                    <tr
                      key={i}
                      style={{
                        background: r.status === "failed" ? "rgba(239,68,68,0.08)" : "transparent",
                        borderTop: "1px solid var(--color-border)",
                      }}
                    >
                      <td className="px-4 py-3 truncate max-w-xs" style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem" }}>
                        {r.file_path}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className="text-xs font-bold px-2 py-0.5 rounded"
                          style={{
                            color: r.status === "completed" ? "var(--color-success)" : r.status === "failed" ? "var(--color-error)" : "var(--color-text-muted)",
                            background: "var(--color-surface-2)",
                          }}
                        >
                          {r.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-bold">{r.chunk_count}</td>
                      <td className="px-4 py-3 text-xs" style={{ color: "var(--color-error)" }}>
                        {r.error_message ?? ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color?: string }) {
  return (
    <div className="rounded-xl p-5 text-center" style={{ background: "var(--color-surface)" }}>
      <div className="text-2xl font-bold mb-1" style={{ color: color ?? "var(--color-text)" }}>
        {value}
      </div>
      <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>{label}</div>
    </div>
  );
}
