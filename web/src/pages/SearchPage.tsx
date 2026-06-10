import { useEffect, useState } from "react";
import type { CollectionInfo, SearchData } from "../api/types";
import { api } from "../api/client";
import SearchResult from "../components/SearchResult";
import { Skeleton } from "../components/Skeleton";
import EmptyState from "../components/EmptyState";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [collection, setCollection] = useState("default");
  const [mode, setMode] = useState<"hybrid" | "dense" | "sparse">("hybrid");
  const [topK, setTopK] = useState(10);
  const [rerank, setRerank] = useState(false);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [result, setResult] = useState<SearchData | null>(null);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  useEffect(() => {
    api.collections.list().then(setCollections).catch(() => {});
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    api.search({ query: query.trim(), collection, mode, top_k: topK, rerank })
      .then(setResult)
      .catch(() => setResult(null))
      .finally(() => setLoading(false));
  };

  const modes: Array<{ value: "hybrid" | "dense" | "sparse"; label: string }> = [
    { value: "hybrid", label: "Hybrid" },
    { value: "dense", label: "Dense" },
    { value: "sparse", label: "Sparse" },
  ];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Search</h1>

      <form onSubmit={handleSubmit} className="mb-8 space-y-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search your knowledge base..."
          className="w-full h-12 px-5 rounded-lg text-sm outline-none focus:ring-2"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text)",
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

          <div className="flex rounded-lg overflow-hidden" style={{ border: "1px solid var(--color-border)" }}>
            {modes.map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setMode(m.value)}
                className="px-4 py-2 text-xs font-bold transition-colors duration-150 cursor-pointer"
                style={{
                  background: mode === m.value ? "var(--color-accent)" : "var(--color-surface)",
                  color: mode === m.value ? "white" : "var(--color-text-muted)",
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          <label className="flex items-center gap-2 text-sm" style={{ color: "var(--color-text-muted)" }}>
            Top-K
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              min={1}
              max={100}
              className="w-16 h-10 px-2 rounded-lg text-sm text-center outline-none"
              style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: "var(--color-text-muted)" }}>
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
              className="w-4 h-4 rounded accent-blue-500"
            />
            Rerank
          </label>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="ml-auto px-6 h-10 rounded-lg text-sm font-bold text-white transition-colors duration-150 cursor-pointer disabled:opacity-50"
            style={{ background: "var(--color-accent)" }}
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </div>
      </form>

      {loading && (
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="rounded-xl p-5" style={{ background: "var(--color-surface)" }}>
              <Skeleton className="h-4 w-2/3 mb-3" />
              <Skeleton className="h-3 w-full mb-2" />
              <Skeleton className="h-3 w-1/2" />
            </div>
          ))}
        </div>
      )}

      {!loading && searched && result && result.results.length === 0 && (
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-12 h-12">
              <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z" />
            </svg>
          }
          title="No results found"
          description="Try different keywords or a broader search query."
        />
      )}

      {!loading && result && result.results.length > 0 && (
        <>
          <div className="space-y-4">
            {result.results.map((r) => (
              <SearchResult key={r.id} result={r} />
            ))}
          </div>
          <div className="mt-6 text-xs" style={{ color: "var(--color-text-muted)" }}>
            {result.total} results in {result.latency_ms.toFixed(0)}ms &middot; {result.mode} mode
          </div>
        </>
      )}
    </div>
  );
}
