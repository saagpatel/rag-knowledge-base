import type {
  AskData,
  AskRequest,
  CollectionInfo,
  DocumentInfo,
  DocumentListData,
  HealthData,
  IngestData,
  IngestRequest,
  JobData,
  MetricsData,
  QueryListData,
  SearchData,
  SearchRequest,
  StatsData,
  SuccessResponse,
} from "./types";

export class ApiError extends Error {
  code: string;
  statusCode: number;
  details?: unknown;

  constructor(code: string, message: string, statusCode: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const body = await res.json();

  if (!body.success) {
    const err = body.error;
    throw new ApiError(
      err?.code ?? "UNKNOWN",
      err?.message ?? res.statusText,
      err?.statusCode ?? res.status,
      err?.details,
    );
  }

  return (body as SuccessResponse<T>).data;
}

export async function* askStream(body: AskRequest): AsyncGenerator<string> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true }),
  });

  if (!res.ok || !res.body) {
    throw new ApiError("STREAM_ERROR", "Failed to start stream", res.status);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data: ")) continue;
        const data = trimmed.slice(6);
        if (data === "[DONE]") return;
        yield data;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export const api = {
  health: () => request<HealthData>("/api/health"),

  search: (body: SearchRequest) =>
    request<SearchData>("/api/search", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  ask: (body: AskRequest) =>
    request<AskData>("/api/ask", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  askStream,

  collections: {
    list: () => request<CollectionInfo[]>("/api/collections"),

    get: (name: string) => request<CollectionInfo>(`/api/collections/${encodeURIComponent(name)}`),

    create: (name: string) =>
      request<CollectionInfo>("/api/collections", {
        method: "POST",
        body: JSON.stringify({ name }),
      }),

    delete: (name: string) =>
      request<{ message: string }>(`/api/collections/${encodeURIComponent(name)}`, {
        method: "DELETE",
      }),
  },

  ingest: (body: IngestRequest) =>
    request<IngestData>("/api/ingest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  documents: {
    list: (params?: { collection?: string; status?: string; limit?: number; offset?: number }) => {
      const qs = new URLSearchParams();
      if (params?.collection) qs.set("collection", params.collection);
      if (params?.status) qs.set("status", params.status);
      if (params?.limit) qs.set("limit", String(params.limit));
      if (params?.offset) qs.set("offset", String(params.offset));
      const s = qs.toString();
      return request<DocumentListData>(`/api/documents${s ? `?${s}` : ""}`);
    },
    get: (id: string) => request<DocumentInfo>(`/api/documents/${encodeURIComponent(id)}`),
    delete: (id: string) =>
      request<{ message: string }>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  },

  jobs: {
    get: (id: string) => request<JobData>(`/api/jobs/${encodeURIComponent(id)}`),
  },

  metrics: () => request<MetricsData>("/api/metrics"),

  stats: (days?: number) => request<StatsData>(`/api/stats${days ? `?days=${days}` : ""}`),

  queries: (params?: { limit?: number; offset?: number; collection?: string; interface?: string; query_type?: string }) => {
    const qs = new URLSearchParams();
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    if (params?.collection) qs.set("collection", params.collection);
    if (params?.interface) qs.set("interface", params.interface);
    if (params?.query_type) qs.set("query_type", params.query_type);
    const s = qs.toString();
    return request<QueryListData>(`/api/queries${s ? `?${s}` : ""}`);
  },
};
