import type { SearchResultItem } from "../api/types";

interface Props {
  result: SearchResultItem;
}

export default function SearchResult({ result }: Props) {
  const scorePercent = Math.round(result.score * 100);

  return (
    <div
      className="relative rounded-xl p-5 transition-colors duration-150"
      style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)" }}
    >
      <div
        className="absolute top-4 right-4 text-xs font-bold px-2 py-1 rounded"
        style={{ background: "var(--color-surface-2)", color: "var(--color-accent)" }}
      >
        {scorePercent}%
      </div>

      <pre
        className="text-sm leading-relaxed mb-4 whitespace-pre-wrap overflow-hidden"
        style={{
          fontFamily: "var(--font-mono)",
          color: "var(--color-text)",
          maxHeight: "4.5em",
          display: "-webkit-box",
          WebkitLineClamp: 3,
          WebkitBoxOrient: "vertical",
        }}
      >
        {result.content}
      </pre>

      <div className="flex items-center gap-3 text-xs" style={{ color: "var(--color-text-muted)" }}>
        <span className="truncate max-w-xs" title={result.file_path}>
          {result.file_path}
        </span>
        <span
          className="px-2 py-0.5 rounded"
          style={{ background: "var(--color-surface-2)" }}
        >
          {result.file_type}
        </span>
        <span>
          chunk {result.chunk_index + 1}/{result.total_chunks}
        </span>
        {result.reranked && (
          <span style={{ color: "var(--color-accent)" }}>reranked</span>
        )}
      </div>
    </div>
  );
}
