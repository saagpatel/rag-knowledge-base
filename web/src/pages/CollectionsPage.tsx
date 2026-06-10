import { useEffect, useState } from "react";
import type { CollectionInfo } from "../api/types";
import { api, ApiError } from "../api/client";
import { SkeletonCard } from "../components/Skeleton";
import StatusBadge from "../components/StatusBadge";
import EmptyState from "../components/EmptyState";

export default function CollectionsPage() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCollections = () => {
    setLoading(true);
    api.collections.list()
      .then(setCollections)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to fetch"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchCollections(); }, []);

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setError(null);
    api.collections.create(newName.trim())
      .then(() => {
        setNewName("");
        fetchCollections();
      })
      .catch((err: unknown) => setError(err instanceof ApiError ? err.message : "Failed to create"))
      .finally(() => setCreating(false));
  };

  const handleDelete = (name: string) => {
    if (!window.confirm(`Delete collection "${name}"? This cannot be undone.`)) return;
    api.collections.delete(name)
      .then(fetchCollections)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to delete"));
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-8">Collections</h1>

      <form onSubmit={handleCreate} className="flex gap-3 mb-8">
        <input
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="Collection name..."
          className="flex-1 h-11 px-4 rounded-lg text-sm outline-none focus:ring-2"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text)",
          }}
        />
        <button
          type="submit"
          disabled={creating || !newName.trim()}
          className="px-5 h-11 rounded-lg text-sm font-bold text-white transition-colors duration-150 cursor-pointer disabled:opacity-50"
          style={{ background: "var(--color-accent)" }}
        >
          {creating ? "Creating..." : "Create"}
        </button>
      </form>

      {error && (
        <div className="rounded-lg px-4 py-3 text-sm mb-6" style={{ background: "var(--color-surface)", color: "var(--color-error)" }}>
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : collections.length === 0 ? (
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="w-12 h-12">
              <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375" />
            </svg>
          }
          title="No collections yet"
          description="Create your first collection to start organizing and searching your documents."
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {collections.map((c) => (
            <div
              key={c.name}
              className="rounded-xl p-6"
              style={{ background: "var(--color-surface)", border: "1px solid var(--color-border)" }}
            >
              <div className="flex items-start justify-between mb-4">
                <h3 className="font-bold text-lg">{c.name}</h3>
                <StatusBadge status={c.status} />
              </div>
              <div className="space-y-2 text-sm mb-5" style={{ color: "var(--color-text-muted)" }}>
                <div className="flex justify-between">
                  <span>Points</span>
                  <span className="font-bold" style={{ color: "var(--color-text)" }}>{c.points_count.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Vectors</span>
                  <span className="font-bold" style={{ color: "var(--color-text)" }}>{c.vectors_count.toLocaleString()}</span>
                </div>
              </div>
              <button
                onClick={() => handleDelete(c.name)}
                className="text-xs px-3 py-1.5 rounded cursor-pointer transition-colors duration-150"
                style={{ color: "var(--color-error)", background: "transparent", border: "1px solid var(--color-error)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--color-error)";
                  e.currentTarget.style.color = "white";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.color = "var(--color-error)";
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
