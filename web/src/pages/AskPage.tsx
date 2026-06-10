import { useEffect, useState } from "react";
import type { AskData, CollectionInfo, SourceItem } from "../api/types";
import { api, askStream } from "../api/client";
import { Skeleton } from "../components/Skeleton";

export default function AskPage() {
  const [query, setQuery] = useState("");
  const [collection, setCollection] = useState("default");
  const [model, setModel] = useState("");
  const [topK, setTopK] = useState(5);
  const [stream, setStream] = useState(false);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [result, setResult] = useState<AskData | null>(null);
  const [streamText, setStreamText] = useState("");
  const [loading, setLoading] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  useEffect(() => {
    api.collections.list().then(setCollections).catch(() => {});
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    setStreamText("");
    setSourcesOpen(false);

    try {
      if (stream) {
        let text = "";
        for await (const token of askStream({
          query: query.trim(),
          collection,
          top_k: topK,
          model: model || undefined,
        })) {
          text += token;
          setStreamText(text);
        }
      } else {
        const data = await api.ask({
          query: query.trim(),
          collection,
          top_k: topK,
          model: model || undefined,
          stream: false,
        });
        setResult(data);
      }
    } catch {
      // errors already visible via empty result
    } finally {
      setLoading(false);
    }
  };

  const answerText = stream ? streamText : result?.answer;
  const sources = result?.sources ?? [];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Ask</h1>

      <form onSubmit={handleSubmit} className="mb-8 space-y-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
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

          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="Model (optional)"
            className="h-10 px-3 rounded-lg text-sm outline-none w-44"
            style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
          />

          <label className="flex items-center gap-2 text-sm" style={{ color: "var(--color-text-muted)" }}>
            Top-K
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              min={1}
              max={20}
              className="w-16 h-10 px-2 rounded-lg text-sm text-center outline-none"
              style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
            />
          </label>

          <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: "var(--color-text-muted)" }}>
            <button
              type="button"
              onClick={() => setStream(!stream)}
              className="relative w-10 h-5 rounded-full transition-colors duration-200 cursor-pointer"
              style={{ background: stream ? "var(--color-accent)" : "var(--color-surface-2)" }}
            >
              <span
                className="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform duration-200"
                style={{ transform: stream ? "translateX(20px)" : "translateX(0)" }}
              />
            </button>
            Stream
          </label>

          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="ml-auto px-6 h-10 rounded-lg text-sm font-bold text-white transition-colors duration-150 cursor-pointer disabled:opacity-50"
            style={{ background: "var(--color-accent)" }}
          >
            {loading ? (stream ? "Streaming..." : "Thinking...") : "Ask"}
          </button>
        </div>
      </form>

      {loading && !stream && (
        <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
          <Skeleton className="h-4 w-full mb-3" />
          <Skeleton className="h-4 w-5/6 mb-3" />
          <Skeleton className="h-4 w-2/3" />
        </div>
      )}

      {answerText && (
        <div className="space-y-4">
          <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
            {loading && stream && (
              <div className="flex items-center gap-2 mb-3 text-xs" style={{ color: "var(--color-accent)" }}>
                <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: "var(--color-accent)" }} />
                Streaming...
              </div>
            )}
            <div
              className="text-sm leading-relaxed whitespace-pre-wrap"
              style={{ fontFamily: "var(--font-sans)", maxWidth: "65ch" }}
            >
              {answerText}
            </div>
            {result && !stream && (
              <div className="mt-4 pt-4 flex gap-4 text-xs" style={{ borderTop: "1px solid var(--color-border)", color: "var(--color-text-muted)" }}>
                <span>Model: {result.model}</span>
                <span>{result.latency_ms.toFixed(0)}ms</span>
                <span>{result.context_chunks_used} chunks used</span>
              </div>
            )}
          </div>

          {stream && !loading && (
            <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>
              Sources not available in streaming mode.
            </p>
          )}

          {sources.length > 0 && (
            <div>
              <button
                onClick={() => setSourcesOpen(!sourcesOpen)}
                className="flex items-center gap-2 text-sm font-bold cursor-pointer mb-3"
                style={{ color: "var(--color-text-muted)" }}
              >
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  className="w-4 h-4 transition-transform duration-150"
                  style={{ transform: sourcesOpen ? "rotate(90deg)" : "rotate(0)" }}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="m8.25 4.5 7.5 7.5-7.5 7.5" />
                </svg>
                {sources.length} Sources
              </button>
              {sourcesOpen && (
                <div className="space-y-2">
                  {sources.map((s, i) => (
                    <SourceCard key={i} source={s} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SourceCard({ source }: { source: SourceItem }) {
  return (
    <div
      className="rounded-lg p-4 text-sm flex items-center justify-between"
      style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)" }}
    >
      <div className="flex items-center gap-3">
        <span className="truncate max-w-xs" title={source.file_path}>{source.file_path}</span>
        <span className="px-2 py-0.5 rounded text-xs" style={{ background: "var(--color-surface-2)" }}>{source.file_type}</span>
      </div>
      <div className="flex items-center gap-3 text-xs" style={{ color: "var(--color-text-muted)" }}>
        <span>chunk {source.chunk_index + 1}/{source.total_chunks}</span>
        <span className="font-bold" style={{ color: "var(--color-accent)" }}>{Math.round(source.score * 100)}%</span>
      </div>
    </div>
  );
}
