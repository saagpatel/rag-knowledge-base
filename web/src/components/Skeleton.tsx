interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className = "h-4 w-full" }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse rounded ${className}`}
      style={{ background: "var(--color-surface-2)" }}
    />
  );
}

export function SkeletonCard() {
  return (
    <div className="rounded-xl p-6" style={{ background: "var(--color-surface)" }}>
      <Skeleton className="h-5 w-1/3 mb-4" />
      <Skeleton className="h-4 w-full mb-2" />
      <Skeleton className="h-4 w-2/3" />
    </div>
  );
}
