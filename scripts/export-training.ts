// Exports the accumulated judgment data as JSONL for offline training and evaluation (e.g. fine-tuning
// a TypeSafe judgment model): one row per model-run analysis, with the input state the models judged
// (rebuilt through the same loader analyzeArticle used), the teacher labels from analyses.output, the
// raw provider responses from receipts (choice probabilities included) and the downstream decision.
// The pipeline keeps writing rows; re-run this any time — it never calls a model.
// Usage: node --env-file=.env scripts/export-training.ts --out .data/training.jsonl [--since 2026-03-01] [--limit 20000]
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { parseArgs } from "node:util";
import { REPO_ROOT } from "@aihot/backend/config";
import { closeDb, sql } from "@aihot/backend/db";
import { loadAnalyzeInput } from "@aihot/backend/editorial/input";

const { values } = parseArgs({
  options: {
    out: { type: "string", default: ".data/training.jsonl" },
    since: { type: "string" },
    limit: { type: "string", default: "20000" },
  },
});

interface RunRow {
  id: number; article_id: string; input_revision: number; prompt_version: string; receipt_ids: number[];
  relevance: string; category: string | null; tags: string[] | null; subjects: string[] | null;
  title_zh: string | null; summary_zh: string | null; reason_zh: string | null;
  score: number | null; selected: boolean; output: Record<string, any> | null; created_at: Date;
}

const runs = await sql<RunRow[]>`
  SELECT id, article_id, input_revision, prompt_version, receipt_ids, relevance, category, tags, subjects,
         title_zh, summary_zh, reason_zh, score, selected, output, created_at
  FROM analyses
  WHERE origin = 'model' ${values.since ? sql`AND created_at >= ${new Date(values.since)}` : sql``}
  ORDER BY id LIMIT ${Number(values.limit)}`;

const articleIds = [...new Set(runs.map((r) => r.article_id))];
const published = new Set(
  articleIds.length
    ? (await sql<{ article_id: string }[]>`SELECT article_id FROM publications WHERE article_id IN ${sql(articleIds)}`).map((p) => p.article_id)
    : [],
);

const lines: string[] = [];
let skipped = 0;
for (const r of runs) {
  // The input text lives on the article, not the analysis; a re-judged article carries a newer
  // revision, so the flag lets the consumer drop rows whose text has moved on.
  const input = await loadAnalyzeInput(r.article_id);
  if (!input) { skipped += 1; continue; }
  const receipts = r.receipt_ids.length
    ? await sql<{ id: number; purpose: string; service: string; model: string; status: string; response: unknown }[]>`
        SELECT id, purpose, service, model, status, response FROM receipts WHERE id IN ${sql(r.receipt_ids)} ORDER BY id`
    : [];
  const out = r.output ?? {};
  lines.push(JSON.stringify({
    caseId: `analysis-${r.id}`,
    createdAt: r.created_at,
    input: {
      ...input,
      revision: input.revision,
      inputRevision: r.input_revision,
      inputCurrent: input.revision === r.input_revision,
    },
    labels: {
      promptVersion: r.prompt_version,
      prefilter: out.prefilter ?? null,
      scores: out.scores ?? null,
      scoreModel: out.scoreModel ?? null,
      threshold: out.threshold ?? null,
      structure: { category: r.category, tags: r.tags, subjects: r.subjects, fact: out.fact ?? null },
      writing: { titleZh: r.title_zh, summaryZh: r.summary_zh, reasonZh: r.reason_zh },
    },
    decision: { relevance: r.relevance, score: r.score, selected: r.selected, published: published.has(r.article_id) },
    receipts: receipts.map((t) => ({ id: t.id, purpose: t.purpose, service: t.service, model: t.model, status: t.status, response: t.response })),
  }));
}

const abs = path.resolve(REPO_ROOT, values.out!);
mkdirSync(path.dirname(abs), { recursive: true });
writeFileSync(abs, lines.join("\n") + (lines.length ? "\n" : ""), { mode: 0o600 });
console.log(`导出 ${lines.length} 条分析（跳过 ${skipped} 条缺材料）→ ${abs}`);
await closeDb();
