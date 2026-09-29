// TypeSafe System One models (Jev): judgment models that return typed choices with probabilities
// instead of generating text. The prefilter step runs one when PREFILTER_MODEL names a TypeSafe
// preset: its PASS/BLOCK/UNKNOWN call is a closed three-way choice, and prefilter is the highest
// volume call in the pipeline. Every other step needs generation or calibrated taste and stays on
// chat models (chatJson refuses the service). The prefilter prompt stays the single rubric: it is
// sent as the question's instructions, so editing industry/prompts/prefilter.md changes both paths.
import { z } from "zod";
import { config, credential } from "../config.ts";
import { sha256 } from "../lib/ids.ts";
import { ModelOutputError, type ModelSpec } from "./llm.ts";
import { paidRequest, ProviderRejectedError, rejectReceivedResponse } from "./receipts.ts";

/** A low-confidence BLOCK becomes UNKNOWN: blocking is the only way an item is lost. */
const BLOCK_MIN_CONFIDENCE = 0.7;

const AnswerSchema = z.object({
  type: z.literal("choice"),
  choice: z.string(),
  probabilities: z.record(z.string(), z.number()),
  confidence: z.number(),
});

const ResponseSchema = z.object({
  model: z.string(),
  answers: z.record(z.string(), AnswerSchema),
  usage: z.record(z.string(), z.number()).nullish(),
});

export type PrefilterLabel = "PASS" | "BLOCK" | "UNKNOWN";

export interface JevPrefilterCall {
  spec: ModelSpec;
  /** The material, as the same text the chat prefilter sees. */
  state: string;
  /** The resolved prefilter prompt; the rubric the choice question points at. */
  standard: string;
  promptVersion: string;
  subject: string;
  attemptTag?: string;
}

export interface JevPrefilterResult {
  label: PrefilterLabel;
  reason: string;
  receiptId: number;
  reused: boolean;
  model: string;
}

function isConnectFailure(error: unknown): boolean {
  const code = (error as { cause?: { code?: string } })?.cause?.code ?? (error as { code?: string })?.code;
  return ["ECONNREFUSED", "ENOTFOUND", "EAI_AGAIN", "UND_ERR_CONNECT_TIMEOUT", "ECONNRESET_BEFORE_SEND", "CERT_HAS_EXPIRED"].includes(code ?? "");
}

export async function jevPrefilter(opts: JevPrefilterCall): Promise<JevPrefilterResult> {
  if (!config.modelCallsEnabled) throw new Error("Model calls are disabled (MODEL_CALLS_ENABLED=false)");
  const baseUrl = credential("models", opts.spec.baseUrlEnv);
  const apiKey = credential("models", opts.spec.apiKeyEnv);
  if (!baseUrl || !apiKey || !opts.spec.model) throw new Error(`Model ${opts.spec.key} is not configured (${opts.spec.baseUrlEnv}, ${opts.spec.apiKeyEnv})`);

  const identity = {
    model: opts.spec.model,
    promptVersion: opts.promptVersion,
    standard: sha256(opts.standard),
    state: sha256(opts.state),
    task: "prefilter-choice",
  };
  const receipt = await paidRequest(
    {
      service: opts.spec.service,
      model: opts.spec.model,
      purpose: "prefilter_article",
      subject: opts.subject,
      identity,
      requestSummary: { promptVersion: opts.promptVersion, standardHash: identity.standard, stateHash: identity.state, stateChars: opts.state.length },
      attemptTag: opts.attemptTag,
    },
    async () => {
      const started = Date.now();
      let res: Response;
      try {
        res = await fetch(`${baseUrl.replace(/\/$/, "")}/systemone`, {
          method: "POST",
          headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
          body: JSON.stringify({
            model: opts.spec.model,
            state: opts.state,
            questions: {
              relevance: {
                type: "choice",
                instructions: {
                  selection_standard: opts.standard,
                  question: "Read `selection_standard` as this site's filtering standard, then decide which category the state belongs to.",
                },
                criteria: { PASS: null, BLOCK: null, UNKNOWN: null },
              },
            },
          }),
          signal: AbortSignal.timeout(120_000),
        });
      } catch (error) {
        if (isConnectFailure(error)) throw new ProviderRejectedError(`connect failed: ${String(error)}`, null, true);
        throw error;
      }
      const text = await res.text();
      if (!res.ok) {
        const retryable = res.status === 429 || res.status >= 500; // 529 (overloaded) is TypeSafe's retry-me
        throw new ProviderRejectedError(`HTTP ${res.status}: ${text.slice(0, 500)}`, res.status, retryable);
      }
      let json: Record<string, unknown>;
      try {
        json = JSON.parse(text);
      } catch {
        json = { unparsable: text.slice(0, 20000) };
      }
      const usage = (json.usage as Record<string, unknown> | undefined) ?? null;
      return { response: { ...json, _latencyMs: Date.now() - started }, requestId: null, usage, cost: null };
    },
  );

  const parsed = ResponseSchema.safeParse(receipt.response);
  if (!parsed.success) {
    await rejectReceivedResponse(receipt.receiptId, `unusable output: ${String(parsed.error).slice(0, 300)}`);
    throw new ModelOutputError(`Model ${opts.spec.key} returned an unusable TypeSafe answer for ${opts.subject}`);
  }
  const answer = parsed.data.answers.relevance;
  const label = answer.choice.trim().toUpperCase();
  if (label !== "PASS" && label !== "BLOCK" && label !== "UNKNOWN") {
    await rejectReceivedResponse(receipt.receiptId, `unknown choice ${answer.choice.slice(0, 50)}`);
    throw new ModelOutputError(`Model ${opts.spec.key} answered ${answer.choice.slice(0, 50)} for ${opts.subject}`);
  }
  const final = label === "BLOCK" && answer.confidence < BLOCK_MIN_CONFIDENCE ? "UNKNOWN" : label;
  const p = (k: string) => Math.round((answer.probabilities[k] ?? 0) * 100);
  const reason = `jev PASS=${p("PASS")}% BLOCK=${p("BLOCK")}% UNKNOWN=${p("UNKNOWN")}% conf=${Math.round(answer.confidence * 100)}%`;
  return { label: final, reason: reason.slice(0, 200), receiptId: receipt.receiptId, reused: receipt.reused, model: opts.spec.key };
}
