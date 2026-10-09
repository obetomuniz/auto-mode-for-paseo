import { readContext, type ContextEntry } from "./route-context";
import { ROUTE_QUESTIONS, parseRouteAnswers, type RouteAnswers } from "./classifier";
import { presetQuestions } from "./preset-classification";
import type { Preset, ProviderSettings } from "../shared/settings";
import { workspaceStateSchema, type WorkspaceState } from "./workspace-state";

const DECISIONS_URL = "https://api.openai.com/v1/decisions";
export const OPENAI_DEFAULT_MODEL = "gpt-6-luna";

export function openAiKey(settings: Pick<ProviderSettings, "openaiApiKey">): string {
  const apiKey = settings.openaiApiKey.trim() || process.env.OPENAI_API_KEY?.trim() || "";
  if (!apiKey) throw new Error("Configure the OpenAI key for the Decisions API in Settings > Plugins > Auto Mode for Paseo, or set OPENAI_API_KEY on the daemon.");
  return apiKey;
}

export async function evaluateOpenAiRoute(input: {
  apiKey: string;
  model: string;
  prompt: string;
  context?: ContextEntry[];
  presets?: readonly Preset[];
  workspace?: WorkspaceState;
}): Promise<RouteAnswers> {
  const roster = presetQuestions(input.presets);
  const body = await askDecisions({
    apiKey: input.apiKey,
    model: input.model,
    state: {
      request: input.prompt,
      ...(input.workspace ? { workspace: workspaceStateSchema.parse(input.workspace) } : {}),
      ...(input.context?.length ? { recentConversation: readContext(input.context).map(({ role, text }) => ({ role, text })) } : {}),
    },
    questions: { ...ROUTE_QUESTIONS, ...roster.questions },
  });
  return parseRouteAnswers(body, roster.ids);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function text(value: unknown, id: string): string {
  if (typeof value === "string" && value.trim()) return value;
  throw new Error(`Decisions question '${id}' has no text.`);
}

/** Translate the shared question definitions. The state keys named in them stay unchanged. */
export function decisionQuestions(questions: Record<string, unknown>): Record<string, unknown>[] {
  return Object.entries(questions).map(([name, question]) => {
    if (!isRecord(question)) throw new Error(`Decisions question '${name}' is invalid.`);
    const instructions = isRecord(question.instructions)
      ? `${text(question.instructions.question, name)} ${text(question.instructions.focus, name)}`
      : text(question.instructions, name);
    if (question.type === "noul") return { type: "predicate", name, instructions };
    if (question.type !== "choice" || !isRecord(question.criteria)) throw new Error(`Decisions question '${name}' has an unsupported type.`);
    const choices = Object.entries(question.criteria).map(([value, criterion]) => {
      if (!isRecord(criterion)) return { value, description: text(criterion, name) };
      const examples = Array.isArray(criterion.examples) ? criterion.examples.map((example) => `"${text(example, name)}"`) : [];
      return { value, description: examples.length ? `${text(criterion.what, name)}. Examples: ${examples.join("; ")}` : text(criterion.what, name) };
    });
    return { type: "choice", name, instructions, choices };
  });
}

/** Return the answer shape that the shared validators read. They check every value. */
function sharedAnswers(body: unknown): { answers: Record<string, unknown> } {
  if (!isRecord(body) || !Array.isArray(body.answers)) throw new Error("Classifier response was missing answers.");
  const answers: Record<string, unknown> = {};
  for (const answer of body.answers) {
    if (!isRecord(answer) || typeof answer.name !== "string") throw new Error("Classifier returned an answer without a name.");
    if (Object.hasOwn(answers, answer.name)) throw new Error(`Classifier answered '${answer.name}' more than once.`);
    if (answer.type === "predicate") {
      answers[answer.name] = { type: "noul", noul: answer.probability };
    } else if (answer.type === "choice") {
      if (!Array.isArray(answer.probabilities)) throw new Error(`Classifier answer '${answer.name}' was not a choice.`);
      const probabilities: Record<string, unknown> = {};
      for (const item of answer.probabilities) {
        if (!isRecord(item) || typeof item.value !== "string") throw new Error(`Classifier answer '${answer.name}' was not a choice.`);
        probabilities[item.value] = item.probability;
      }
      answers[answer.name] = { type: "choice", choice: answer.choice, probabilities, confidence: answer.confidence };
    }
  }
  return { answers };
}

/** One Decisions API call. Callers own the state and questions they send. */
export async function askDecisions(input: { apiKey: string; model: string; state: Record<string, unknown>; questions: Record<string, unknown> }): Promise<{ answers: Record<string, unknown> }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 20_000);
  let response: Response;
  try {
    response = await fetch(DECISIONS_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${input.apiKey}`,
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify({ model: input.model, input: JSON.stringify(input.state), questions: decisionQuestions(input.questions) }),
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error("OpenAI Decisions API timed out after 20s.");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }

  const raw = await response.text();
  let body: unknown;
  try {
    body = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`OpenAI Decisions API returned non-JSON (${response.status}).`);
  }

  if (!response.ok) {
    const detail = isRecord(body) && isRecord(body.error) && typeof body.error.message === "string" ? body.error.message : raw.slice(0, 200);
    if (response.status === 401) {
      throw new Error("OpenAI API key rejected. Set it in plugin settings or OPENAI_API_KEY.");
    }
    throw new Error(`OpenAI Decisions API ${response.status}: ${detail || "request failed"}`);
  }

  return sharedAnswers(body);
}
