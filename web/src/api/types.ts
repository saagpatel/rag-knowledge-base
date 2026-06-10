/** TypeScript interfaces mirroring the Pydantic schemas in api/schemas.py */

// --- Envelope ---

export interface Meta {
  request_id: string;
  timestamp: string;
}

export interface SuccessResponse<T> {
  success: true;
  data: T;
  meta: Meta;
}

export interface ErrorDetail {
  code: string;
  message: string;
  statusCode: number;
  details?: unknown;
}

export interface ErrorResponse {
  success: false;
  error: ErrorDetail;
  meta: Meta;
}

export type ApiResponse<T> = SuccessResponse<T> | ErrorResponse;

// --- Health ---

export interface ServiceCheck {
  status: string;
  detail: string | null;
}

export interface HealthData {
  status: string;
  ollama: ServiceCheck;
  qdrant: ServiceCheck;
  sqlite: ServiceCheck;
  uptime_seconds: number;
  version: string;
}

// --- Ingest ---

export interface IngestRequest {
  path: string;
  collection?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  patterns?: string[] | null;
}

export interface FileIngestResult {
  file_path: string;
  status: string;
  chunk_count: number;
  error_message: string | null;
}

export interface IngestData {
  total_files: number;
  processed: number;
  failed: number;
  skipped: number;
  results: FileIngestResult[];
}

// --- Search ---

export interface SearchRequest {
  query: string;
  collection?: string;
  mode?: "hybrid" | "dense" | "sparse";
  top_k?: number;
  rerank?: boolean;
  filters?: Record<string, unknown> | null;
}

export interface SearchResultItem {
  id: string | number;
  score: number;
  content: string;
  file_path: string;
  file_type: string;
  chunk_index: number;
  total_chunks: number;
  reranked: boolean;
}

export interface SearchData {
  results: SearchResultItem[];
  total: number;
  query: string;
  collection: string;
  mode: string;
  latency_ms: number;
}

// --- Ask ---

export interface AskRequest {
  query: string;
  collection?: string;
  mode?: string;
  top_k?: number;
  model?: string | null;
  stream?: boolean;
}

export interface SourceItem {
  file_path: string;
  score: number;
  chunk_index: number;
  total_chunks: number;
  file_type: string;
}

export interface AskData {
  answer: string;
  sources: SourceItem[];
  query: string;
  model: string;
  latency_ms: number;
  context_chunks_used: number;
}

// --- Collections ---

export interface CollectionInfo {
  name: string;
  points_count: number;
  vectors_count: number;
  status: string;
}

export interface CreateCollectionRequest {
  name: string;
}

// --- Documents ---

export interface DocumentInfo {
  id: string;
  collection_id: string;
  filename: string;
  file_path: string;
  file_type: string;
  file_hash: string;
  chunk_count: number;
  status: string;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentListData {
  documents: DocumentInfo[];
  total: number;
  collection: string | null;
}

// --- Analytics ---

export interface QueryRecord {
  id: string;
  query_text: string;
  query_type: string;
  search_mode: string;
  result_count: number;
  latency_ms: number;
  interface: string;
  created_at: string;
}

export interface QueryListData {
  queries: QueryRecord[];
  total: number;
}

export interface StatsData {
  total_queries: number;
  avg_latency_ms: number;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  queries_by_interface: Record<string, number>;
  queries_by_type: Record<string, number>;
  top_collections: Array<{ name: string; count: number }>;
  period_days: number;
}

// --- Jobs ---

export interface JobData {
  job_id: string;
  status: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  started_at: string | null;
  completed_at: string | null;
}

// --- Metrics ---

export interface MetricsData {
  total_queries: number;
  latency_p50: number;
  latency_p95: number;
  latency_p99: number;
  cache_hit_rate: number;
  cache_size: number;
  active_jobs: number;
}
