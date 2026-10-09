import assert from "node:assert/strict";
import { test } from "node:test";
import { askDecisions, decisionQuestions, evaluateOpenAiRoute, openAiKey } from "../server/openai-decisions";
import { ROUTE_QUESTIONS, pickIntent, type ChoiceAnswer } from "../server/classifier";
import { presetQuestions } from "../server/preset-classification";
import { CONTEXT_MESSAGES, CONTEXT_TEXT_LIMIT } from "../server/route-context";
import { classifyPrompt, routePrompt } from "../server/routing";
import { detectTaskTypes, SCOPE_TYPE_QUESTIONS } from "../server/scope-types";
import { parseWorkspaceState } from "../server/workspace-state";
import { defaults } from "../shared/settings";
import { answers } from "./fixtures";

const choice = (name: string, answer: ChoiceAnswer) => ({ type: "choice", name, choice: answer.choice, confidence: answer.confidence,
  probabilities: Object.entries(answer.probabilities).map(([value, probability]) => ({ value, probability })) });

// The Decisions API returns an array of named answers with probability arrays.
function decisions(result = answers()) {
  const { presetScores, ...rest } = result;
  return { answers: [
    ...Object.entries(rest).map(([name, answer]) => choice(name, answer)),
    ...presetQuestions(defaults.presets).ids.map((id, index) => ({ type: "predicate", name: `preset_${index}`, probability: presetScores?.[id] ?? 0 })),
  ] };
}

test("OpenAI classification sends the bounded state and translated questions to the Decisions API", async (t) => {
  const context = Array.from({ length: CONTEXT_MESSAGES + 2 }, (_, index) => ({ id: `m${index}`, role: "user" as const, text: `${index}-`.padEnd(CONTEXT_TEXT_LIMIT + 500, "x") }));
  const workspace = parseWorkspaceState("2300\t418\tprivate-path\0", "");
  t.mock.method(globalThis, "fetch", async (...[url, init]: Parameters<typeof fetch>) => {
    assert.equal(url, "https://api.openai.com/v1/decisions");
    assert.equal(new Headers(init?.headers).get("authorization"), "Bearer test-key");
    const request = JSON.parse(String(init?.body));
    assert.deepEqual(Object.keys(request).sort(), ["input", "model", "questions"]);
    assert.equal(request.model, "gpt-6-luna");
    const state = JSON.parse(request.input);
    assert.equal(state.request, "Review the change");
    assert.deepEqual(state.workspace, workspace);
    assert.equal(state.recentConversation.length, CONTEXT_MESSAGES);
    assert.ok(state.recentConversation.every((item: { text: string }) => item.text.length <= CONTEXT_TEXT_LIMIT));
    assert.deepEqual(Object.keys(state.recentConversation[0]), ["role", "text"]);
    assert.ok(!request.input.includes("private"));
    const questions: Record<string, any>[] = request.questions;
    assert.deepEqual(questions.map((question) => question.name), [...Object.keys(ROUTE_QUESTIONS), ...defaults.presets.map((_, index) => `preset_${index}`)]);
    const intent = questions.find((question) => question.name === "intent")!;
    assert.equal(intent.type, "choice");
    assert.equal(intent.instructions, `${ROUTE_QUESTIONS.intent.instructions.question} ${ROUTE_QUESTIONS.intent.instructions.focus}`);
    assert.deepEqual(intent.choices.map((item: { value: string }) => item.value), ["discuss", "review", "implement"]);
    assert.match(intent.choices[1].description, /^An explicit audit.*\. Examples: "Review this PR"; /);
    const fit = questions.find((question) => question.name === "preset_0")!;
    assert.deepEqual(Object.keys(fit).sort(), ["instructions", "name", "type"]);
    assert.equal(fit.type, "predicate");
    for (const preset of defaults.presets) {
      for (const excluded of [preset.instructions, preset.model, preset.name]) assert.ok(!JSON.stringify(questions).includes(excluded));
    }
    return Response.json(decisions());
  });
  const extra = { ...workspace, raw: "private diff" };
  assert.deepEqual(await evaluateOpenAiRoute({ apiKey: "test-key", model: "gpt-6-luna", prompt: "Review the change", context, workspace: extra }), answers());
});

test("the question translation rejects definitions it cannot represent", () => {
  assert.deepEqual(decisionQuestions(SCOPE_TYPE_QUESTIONS.jev)[0].choices, Object.entries(SCOPE_TYPE_QUESTIONS.jev.taskType.criteria).map(([value, description]) => ({ value, description })));
  assert.throws(() => decisionQuestions({ depth: { type: "score", instructions: "How deep?" } }), /unsupported type/);
  assert.throws(() => decisionQuestions({ fit: { type: "noul", instructions: " " } }), /no text/);
  assert.throws(() => decisionQuestions({ fit: "noul" }), /invalid/);
});

test("OpenAI routing selects a preset from predicate scores and never calls another classifier", async (t) => {
  const settings = { ...defaults, classifier: "openai" as const, apiKey: "typesafe-key", openaiApiKey: "openai-key" };
  const urls: string[] = [];
  let fail = false;
  t.mock.method(globalThis, "fetch", async (...[url, init]: Parameters<typeof fetch>) => {
    urls.push(String(url));
    assert.ok(!JSON.stringify(init).includes("typesafe-key"));
    if (fail) return Response.json({ error: { message: "The model is overloaded." } }, { status: 503 });
    return Response.json(decisions(answers({ intent: { type: "choice", choice: "review", probabilities: { review: 0.9, discuss: 0.1 }, confidence: 0.8 } })));
  });
  const route = await routePrompt("Are the current changes good?", [], settings);
  assert.equal(route.classifier, "openai");
  assert.equal(route.presetId, "critic");
  assert.equal(route.intent, "review");
  assert.equal(route.confidence, 0.8);
  assert.equal(route.plan, false);
  assert.equal(route.fast, false);
  fail = true;
  await assert.rejects(routePrompt("Fix it", [], settings), /OpenAI Decisions API 503: The model is overloaded\./);
  assert.deepEqual(new Set(urls), new Set(["https://api.openai.com/v1/decisions"]));
});

test("a missing or rejected OpenAI key stops classification with guidance", async (t) => {
  const saved = process.env.OPENAI_API_KEY;
  t.after(() => { if (saved === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = saved; });
  delete process.env.OPENAI_API_KEY;
  const fetched = t.mock.method(globalThis, "fetch", async () => Response.json({ error: { message: "Incorrect API key provided." } }, { status: 401 }));
  // A stored TypeSafe key is never used for OpenAI.
  await assert.rejects(classifyPrompt("Fix it", [], { ...defaults, classifier: "openai", apiKey: "typesafe-key" }), /Configure the OpenAI key/);
  assert.equal(fetched.mock.callCount(), 0);
  process.env.OPENAI_API_KEY = " env-key ";
  assert.equal(openAiKey({ openaiApiKey: " " }), "env-key");
  assert.equal(openAiKey({ openaiApiKey: "stored-key" }), "stored-key");
  await assert.rejects(classifyPrompt("Fix it", [], { ...defaults, classifier: "openai" }), /OpenAI API key rejected/);
});

test("malformed Decisions responses stop the turn instead of producing a route", async (t) => {
  let body: unknown;
  t.mock.method(globalThis, "fetch", async () => typeof body === "string" ? new Response(body) : Response.json(body));
  const evaluate = () => evaluateOpenAiRoute({ apiKey: "test-key", model: "gpt-6-luna", prompt: "Fix it" });
  const valid = decisions();
  const replace = (name: string, values: Record<string, unknown>) => ({ answers: valid.answers.map((answer) => answer.name === name ? { ...answer, ...values } : answer) });
  const cases: [unknown, RegExp][] = [
    ["<html>", /returned non-JSON \(200\)/],
    [{ answers: { intent: valid.answers[0] } }, /missing answers/],
    [{ answers: [{ type: "choice", choice: "implement" }] }, /without a name/],
    [{ answers: [...valid.answers, valid.answers[0]] }, /more than once/],
    [{ answers: valid.answers.filter((answer) => answer.name !== "intent") }, /'intent' was not a choice/],
    [replace("intent", { probabilities: { implement: 1 } }), /'intent' was not a choice/],
    [replace("intent", { probabilities: [{ value: "implement", probability: 2 }] }), /invalid probability/],
    [replace("intent", { confidence: undefined }), /'intent' was not a choice/],
    [replace("preset_0", { probability: 1.5 }), /'preset_0' was not a noul/],
    [replace("preset_1", { type: "score", score: 0.4 }), /'preset_1' was not a noul/],
    [replace("intent", { choice: "full-access" }), /unknown intent/],
  ];
  for (const [response, error] of cases) {
    body = response;
    await assert.rejects(async () => pickIntent(await evaluate()), error);
  }
});

test("OpenAI scope detection sends only the scope and the fixed task-type question", async (t) => {
  const settings = { ...defaults, classifier: "openai" as const, apiKey: "typesafe-key", openaiApiKey: "secret-key", openaiModel: " ",
    presets: [{ ...defaults.presets[0], instructions: "private instructions" }] };
  const scope = "Translate technical prose into Portuguese.";
  const bodies: string[] = [];
  let taskType = "write";
  t.mock.method(globalThis, "fetch", async (...[url, init]: Parameters<typeof fetch>) => {
    assert.equal(url, "https://api.openai.com/v1/decisions");
    bodies.push(String(init?.body));
    return Response.json({ answers: [choice("taskType", { type: "choice", choice: taskType, probabilities: { [taskType]: 1 }, confidence: 1 })] });
  });
  assert.deepEqual(await detectTaskTypes(`  ${scope}  `, settings), ["write"]);
  const request = JSON.parse(bodies[0]);
  assert.equal(request.model, "gpt-6-luna");
  assert.deepEqual(JSON.parse(request.input), { scope });
  assert.deepEqual(request.questions, decisionQuestions(SCOPE_TYPE_QUESTIONS.jev));
  for (const value of ["secret-key", "typesafe-key", "private instructions", settings.presets[0].description]) assert.ok(!bodies[0].includes(value));
  taskType = "invalid";
  await assert.rejects(detectTaskTypes(scope, settings), /unknown task type/);
});

test("a Decisions request stops after 20 seconds", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  t.mock.method(globalThis, "fetch", (...[, init]: Parameters<typeof fetch>) => new Promise((_resolve, reject) => {
    init!.signal!.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
  }));
  const pending = askDecisions({ apiKey: "test-key", model: "gpt-6-luna", state: { request: "Fix it" }, questions: SCOPE_TYPE_QUESTIONS.jev });
  t.mock.timers.tick(19_999);
  t.mock.timers.tick(1);
  await assert.rejects(pending, /timed out after 20s/);
});
