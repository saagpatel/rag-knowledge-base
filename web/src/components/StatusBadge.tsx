interface StatusBadgeProps {
  status: string;
  label?: string;
}

const dotColor: Record<string, string> = {
  ok: "var(--color-success)",
  healthy: "var(--color-success)",
  green: "var(--color-success)",
  error: "var(--color-error)",
  degraded: "var(--color-warning)",
};

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  const color = dotColor[status.toLowerCase()] ?? "var(--color-text-muted)";
  const text = label ?? status;

  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <span
        className="w-2 h-2 rounded-full shrink-0"
        style={{ background: color }}
      />
      <span className="capitalize">{text}</span>
    </span>
  );
}
