export type Outcome = "answered" | "escalated" | "pending";

export interface Citation {
  index: number;
  title: string;
  source: string;
  section: string;
  url: string;
  score: number;
  retriever: string;
  components: Record<string, number>;
  snippet: string;
}

export interface TraceEntry {
  seq?: number;
  node: string;
  label?: string;
  event: string;
  ts?: number;
  duration_ms: number;
  detail: Record<string, unknown>;
}

export interface RiskResult {
  patient_id: string;
  risk_score: number;
  band: string;
  outcome_predicted: string;
  model: string;
  matched_on?: string;
  top_features: string[];
  observed_triage: Record<string, number | string | null>;
  model_performance: Record<string, unknown>;
  caveat: string;
}

export interface GraphPath {
  describe: string;
  nodes: string[];
  score: number;
}

export interface CriticReport {
  faithfulness?: number;
  total_claims?: number;
  supported_claims?: number;
  unsupported?: { claim: string; why: string }[];
  reasoning?: string;
  method?: string;
}

export interface QueryResult {
  run_id: string;
  query: string;
  normalized_query?: string;
  answer: string;
  outcome: Outcome;
  escalated: boolean;
  escalation_reason: string;
  citations: Citation[];
  groundedness: number;
  best_groundedness: number;
  critic_report: CriticReport;
  route: string;
  route_confidence: number;
  route_reasoning: string;
  question_type?: string;
  entities: string[];
  attempts: number;
  risk: RiskResult | null;
  graph_paths: GraphPath[];
  sources_reviewed: number;
  contains_identifiers: boolean;
  degraded: boolean;
  trace: TraceEntry[];
  llm_usage: Record<string, unknown>;
  provider: Record<string, unknown>;
}

export interface MetricsSummary {
  chunks_indexed: number;
  graph: { nodes?: number; edges?: number };
  runs: {
    runs_total: number;
    answered: number;
    escalated: number;
    escalation_rate: number;
    avg_groundedness: number;
  };
  llm: Record<string, number | Record<string, number>>;
  degraded: boolean;
  provider: Record<string, unknown>;
  thresholds: { groundedness: number; max_reformulations: number };
}

export interface HealthComponent {
  ready?: boolean;
  degraded?: boolean;
  [k: string]: unknown;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  components: Record<string, HealthComponent>;
  disclaimer: string;
}
