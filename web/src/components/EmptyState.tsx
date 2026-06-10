import type { ReactNode } from "react";

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="mb-4" style={{ color: "var(--color-text-muted)" }}>
        {icon}
      </div>
      <h3 className="text-lg font-bold mb-2">{title}</h3>
      <p className="text-sm mb-6 max-w-sm" style={{ color: "var(--color-text-muted)" }}>
        {description}
      </p>
      {action && (
        <button
          onClick={action.onClick}
          className="px-4 py-2.5 rounded-lg text-sm font-bold text-white transition-colors duration-150 cursor-pointer"
          style={{ background: "var(--color-accent)" }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--color-accent-hover)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "var(--color-accent)")}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
