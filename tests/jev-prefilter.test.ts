// The prefilter on a TypeSafe System One model (Jev): the closed three-way choice runs through
// providers/jev.ts, a confident BLOCK stops the item, a low-confidence BLOCK goes on as UNKNOWN,
// and the rest of the pipeline keeps its chat models. The chat steps are answered by the same stub.
import { Reply, stub, tag } from "./setup.ts";
import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { closeDb, sql } from "@aihot/backend/db";
import { upsertMaterial } from "@aihot/backend/content/materials";
import { analyzeArticle } from "@aihot/backend/editorial/analyze";
import { MODELS } from "@aihot/backend/providers/llm";
import { jevPrefilter } from "@aihot/backend/providers/jev";
import { stopBoss } from "@aihot/backend/jobs/queue";

const T = tag();
const SOURCE = `test-jev-${T}`;

// Choice answers keyed by a marker in the material text/state.
const choices: Record<string, { choice: string; confidence: number }> = {
  JEXPRESS: { choice: "PASS", confidence: 0.97 },
  JHOT: { choice: "BLOCK", confidence: 0.92 },
  JLOW: { choice: "BLOCK", confidence: 0.4 },
  JODD: { choice: "MAYBE", confidence: 0.9 },
};

const provider = await stub((hit, req) => {
  if (req.url.includes("/systemone")) {
    const body = JSON.parse(req.body) as { state: string; questions: Record<string, { type: string }> };
    assert.equal(body.questions.relevance.type, "choice");
    const key = Object.keys(choices).find((m) => body.state.includes(m)) ?? "";
    const a = choices[key];
    if (!a) return new Reply(422, { error: { message: `no stub answer for state` } });
    const probabilities: Record<string, number> = { PASS: 0, BLOCK: 0, UNKNOWN: 0 };
    probabilities[a.choice] = a.confidence;
    for (const k of Object.keys(probabilities)) if (k !== a.choice) probabilities[k] = (1 - a.confidence) / 2;
    return { model: "jev-test", answers: { relevance: { type: "choice", choice: a.choice, probabilities, confidence: a.confidence } }, usage: { input_tokens: 900, output_tokens: 10 } };
  }
  // The chat steps (score, understand, structure, summarize) keep their usual shapes.
  const body = JSON.parse(req.body) as { messages: Array<{ role: string; content: unknown }> };
  const system = String(body.messages[0]?.role === "system" ? body.messages[0]!.content : "");
  const answer = (content: unknown) => ({ id: `stub-${hit}`, model: "stub", choices: [{ message: { content: typeof content === "string" ? content : JSON.stringify(content) } }], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } });
  if (system.includes("事件注意力评分器")) return answer({ attentionScore: 80 });
  if (system.includes("内容理解编辑")) return answer({ itemType: "model_release", authorRole: "principal", tags: ["模型发布"], editorialJudgment: "理由", titleZh: `理解标题 ${T}`, summaryZh: `理解摘要 ${T}。第二句补充一个关键数字。` });
  if (system.includes("资料结构化助手")) return answer({ category: "ai-models", tags: ["模型发布"], subjects: [], fact: { title: "事实", subject: null, action: null, object: null, occurredAt: null } });
  return answer(`title_zh: 翻译标题 ${T}\nsummary_zh: 翻译摘要 ${T}。第二句补充影响。`);
});
process.env.TYPESAFE_BASE_URL = `${provider.url}/v1`;
process.env.TYPESAFE_API_KEY = "test-key";
process.env.PREFILTER_MODEL = "jev-latest";
for (const env of ["DASHSCOPE_BASE_URL", "ZHIPU_BASE_URL", "DEEPSEEK_BASE_URL"]) process.env[env] = `${provider.url}/v1`;
for (const env of ["DASHSCOPE_API_KEY", "ZHIPU_API_KEY", "DEEPSEEK_API_KEY"]) process.env[env] = "test-key";

before(async () => {
  await sql`INSERT INTO sources (id, name, kind, tier, participation_mode, next_fetch_at) VALUES (${SOURCE}, 'Test jev source', 'rss', 'T1', 'editorial', '2100-01-01')`;
});
after(async () => {
  await provider.close();
  await stopBoss();
  await closeDb();
});

const article = async (marker: string) =>
  (await upsertMaterial({
    sourceId: SOURCE, url: `https://example.com/${marker}-${T}`, title: `${marker} model release ${T}`,
    bodyText: `${marker}: a lab released a model with benchmark and pricing details. (${T})`,
    bodyStatus: "ok", via: "fetch", publishedAt: new Date("2026-09-28T01:02:03Z"),
  } as never)).articleId;
const row = async (id: string) =>
  (await sql<{ selected: boolean; relevance: string; score: string | null; title_zh: string | null; output: { prefilter?: { label?: string; reason?: string } } }[]>`
    SELECT selected, relevance, score, title_zh, output FROM analyses WHERE article_id = ${id} ORDER BY id DESC LIMIT 1`)[0]!;

test("a PASS choice runs the usual pipeline, and the prefilter receipt lands on the typesafe service", async () => {
  const id = await article("JEXPRESS");
  const res = await analyzeArticle(id);
  assert.deepEqual([res!.output!.relevance, res!.output!.selected], ["pass", true]);
  const r = await row(id);
  assert.deepEqual(r.output.prefilter?.label, "PASS");
  assert.match(r.output.prefilter?.reason ?? "", /^jev PASS=97% /);
  const [receipt] = await sql<{ service: string; purpose: string; model: string }[]>`
    SELECT service, purpose, model FROM receipts WHERE id = ${res!.receiptIds[0]!}`;
  assert.ok(receipt, "prefilter receipt recorded");
  assert.deepEqual([receipt.service, receipt.purpose, receipt.model], ["typesafe", "prefilter_article", "jev-latest"]);
});

test("a confident BLOCK stops the item before any score", async () => {
  const id = await article("JHOT");
  const res = await analyzeArticle(id);
  assert.deepEqual([res!.output!.relevance, res!.output!.selected, res!.output!.score], ["block", false, null]);
  const r = await row(id);
  assert.deepEqual(r.output.prefilter?.label, "BLOCK");
  assert.match(r.output.prefilter?.reason ?? "", /^jev PASS=4% BLOCK=92% UNKNOWN=4% conf=92%$/);
});

test("a low-confidence BLOCK goes on as UNKNOWN and gets scored", async () => {
  const id = await article("JLOW");
  const res = await analyzeArticle(id);
  assert.deepEqual([res!.output!.relevance, res!.output!.selected], ["pass", true]);
  const r = await row(id);
  assert.deepEqual(r.output.prefilter?.label, "UNKNOWN");
  assert.notEqual(r.score, null);
});

test("the probability trail lands in the stored prefilter reason; an off-list choice fails the receipt", async () => {
  const spec = MODELS["jev-latest"]!;
  const ok = await jevPrefilter({ spec, state: `JEXPRESS direct state ${T}`, standard: "standard", promptVersion: "t1", subject: `jev-test-${T}` });
  assert.match(ok.reason, /^jev PASS=97% /);
  assert.equal(ok.label, "PASS");
  const again = await jevPrefilter({ spec, state: `JEXPRESS direct state ${T}`, standard: "standard", promptVersion: "t1", subject: `jev-test-${T}` });
  assert.equal(again.reused, true, "the same state and standard reuses the receipt");
  await assert.rejects(
    () => jevPrefilter({ spec, state: `JODD direct state ${T}`, standard: "standard", promptVersion: "t1", subject: `jev-test-odd-${T}` }),
    /unusable TypeSafe answer|answered MAYBE/,
  );
});
