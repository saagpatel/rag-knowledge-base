import { useEffect, useState } from "react";
import type { CollectionInfo, DocumentInfo, DocumentListData } from "../api/types";
import { api } from "../api/client";
import { Skeleton } from "../components/Skeleton";
import StatusBadge from "../components/StatusBadge";
import EmptyState from "../components/EmptyState";
import { timeAgo } from "../utils/time";

const PAGE_SIZE = 20;

export default function DocumentsPage() {
  const [data, setData] = useState<DocumentListData | null>(null);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [collection, setCollection] = useState("");
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);

  const fetchDocuments = (col: string, st: string, off: number) => {
    setLoading(true);
    setError(null);
    api.documents
      .list({
        collection: col || undefined,
        status: st || undefined,
        limit: PAGE_SIZE,
        offset: off,
      })
      .then((d) => setData(d))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to fetch documents"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    api.collections.list().then(setCollections).catch(() => {});
  }, []);

  useEffect(() => {
    fetchDocuments(collection, status, offset);
  }, [collection, status, offset]);

  const handleDelete = (doc: DocumentInfo) => {
    if (!window.confirm(`Delete document "${doc.filename}"? This cannot be undone.`)) return;
    api.documents
      .delete(doc.id)
      .then(() => fetchDocuments(collection, status, offset))
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to delete"));
  };

  const total = data?.total ?? 0;
  const showingFrom = total > 0 ? offset + 1 : 0;
  const showingTo = Math.min(offset + PAGE_SIZE, total);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Documents</h1>

      {/* Filter bar */}
      <div className="flex gap-3 mb-6">
        <select
          value={collection}
          onChange={(e) => { setCollection(e.target.value); setOffset(0); }}
          className="h-10 px-3 rounded-lg text-sm outline-none cursor-pointer"
          style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
        >
          <option value="">All collections</option>
          {collections.map((c) => (
            <option key={c.name} value={c.name}>{c.name}</option>
          ))}
        </select>
        <select
          value={status}
          onChange={(e) => { setStatus(e.target.value); setOffset(0); }}
          className="h-10 px-3 rounded-lg text-sm outline-none cursor-pointer"
          style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", color: "var(--color-text)" }}
        >
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
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
      ) : !data || data.documents.length === 0 ? (
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-12 h-12">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
          }
          title="No documents found"
          description="Ingest some files to see them listed here."
        />
      ) : (
        <>
          <div className="overflow-x-auto rounded-xl" style={{ background: "var(--color-surface)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Filename</th>
                  <th className="text-left px-4 py-3 font-bold hidden lg:table-cell" style={{ color: "var(--color-text-muted)" }}>Path</th>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Type</th>
                  <th className="text-right px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Chunks</th>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Status</th>
                  <th className="text-left px-4 py-3 font-bold" style={{ color: "var(--color-text-muted)" }}>Created</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {data.documents.map((doc) => (
                  <tr key={doc.id} style={{ borderBottom: "1px solid var(--color-border)" }}>
                    <td className="px-4 py-3 font-bold">{doc.filename}</td>
                    <td className="px-4 py-3 font-mono text-xs hidden lg:table-cell max-w-48 truncate" style={{ color: "var(--color-text-muted)" }}>
                      {doc.file_path}
                    </td>
                    <td className="px-4 py-3" style={{ color: "var(--color-text-muted)" }}>{doc.file_type}</td>
                    <td className="px-4 py-3 text-right">{doc.chunk_count}</td>
                    <td className="px-4 py-3"><StatusBadge status={doc.status} /></td>
                    <td className="px-4 py-3 text-xs" style={{ color: "var(--color-text-muted)" }}>{timeAgo(doc.created_at)}</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleDelete(doc)}
                        className="text-xs px-2 py-1 rounded cursor-pointer transition-colors duration-150"
                        style={{ color: "var(--color-error)", background: "transparent" }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(239,68,68,0.1)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        Delete
                      </button>
                    </td>
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
