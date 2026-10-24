"use strict";

(() => {
  const API = "/ui-api"; // the server holds the API key; this page never sees one
  const $ = (id) => document.getElementById(id);
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const ANSWER_LABEL = { yes: "Yes", no: "No", not_sure: "Not sure" };
  const RESULT_LABEL = { worked: "fixed", partly: "partly fixed", not_worked: "did not fix" };
  const UPPER_WORDS = new Set(["MG", "EV", "BS4", "BS6", "SUV"]);
  const DEFAULT_MODEL = "MG_COMET_EV"; // only model supported for now; no picker shown

  const PROGRESS_PREFIX = "dtcbot:progress:"; // one saved diagnosis per id, so Resume can restore the exact spot
  const LAST_ID_KEY = "dtcbot:lastId";

  let diagnosisId = null;
  let history = []; // [{fromId, question, answer}] - also the back/forward stack
  let future = []; // [{fromId, question, answer}] popped off history by "Back", replayed by "Forward"
  let lastConclusion = null;
  let lastEvidence = null;
  let vehicle = { dtc: "", model: "" };
  let timer = null;
  let busy = false;
  let inlineBusy = false; // true while only the question card shows a small loader
  let asking = false; // whether the current diagnosis is waiting for an answer

  // ---------- helpers: text only, never parsed as markup ----------
  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    if (props.cls) node.className = props.cls;
    if (props.text !== undefined) node.textContent = props.text;
    for (const [k, v] of Object.entries(props.attrs || {})) node.setAttribute(k, v);
    for (const c of children) node.append(c);
    return node;
  }
  const clear = (node) => node.replaceChildren();
  const show = (node, on = true) => { node.hidden = !on; };
  const friendly = (m) => m.split("_").map((w) => (UPPER_WORDS.has(w) || /\d/.test(w) ? w : w.charAt(0) + w.slice(1).toLowerCase())).join(" ");

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  function setBusy(text, title = "Working on it") {
    busy = true;
    const started = Date.now();
    $("loading-title").textContent = title;
    $("status-text").textContent = text;
    $("status-time").textContent = "0:00";
    show($("status"));
    clearInterval(timer);
    timer = setInterval(() => {
      const t = Math.round((Date.now() - started) / 1000);
      $("status-time").textContent = `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
    }, 1000);
    document.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  }
  function clearBusy() {
    busy = false;
    clearInterval(timer);
    show($("status"), false);
    document.querySelectorAll("button").forEach((b) => { b.disabled = false; });
    updateNavButtons();
  }
  function updateNavButtons() {
    $("back-btn").disabled = history.length === 0;
    $("forward-btn").disabled = future.length === 0;
  }
  // Small loader for follow-up questions: only the question card changes, the rest of the page stays as it is.
  function setInlineBusy(on, text = "Working out the next question…") {
    inlineBusy = on;
    busy = on;
    clearInterval(timer);
    show($("q-loader"), on);
    $("answers").hidden = on || !asking;
    if (on) {
      const started = Date.now();
      $("q-loader-text").textContent = text;
      $("q-loader-time").textContent = "";
      timer = setInterval(() => {
        const s = Math.round((Date.now() - started) / 1000);
        $("q-loader-time").textContent = s >= 2 ? `${s} s` : "";
      }, 1000);
    }
  }
  function showError(msg) { const e = $("error"); e.textContent = msg; show(e); }
  const hideError = () => show($("error"), false);

  function screen(name) {
    for (const id of ["login", "blocked", "start", "work", "done"]) show($(`screen-${id}`), id === name);
    document.body.classList.toggle("on-start", name === "start");
    show($("action-bar"), name === "start" || name === "work");
    show($("start-btn"), name === "start");
    show($("reset"), name === "work");
    show($("back-btn"), name === "work");
    show($("forward-btn"), name === "work");
    if (name === "start") { show($("copy-btn"), false); show($("print-btn"), false); updateStartNote(); }
  }
  function updateStartNote() {
    const note = $("action-note");
    note.classList.remove("strong");
    const desc = $("query-text").value.trim();
    note.textContent = desc ? `Matching your description on ${friendly(DEFAULT_MODEL)}. Ready to start.`
      : "Describe the problem or enter a fault code.";
  }
  const STEP_NAMES = ["Fault code", "Questions", "Diagnosis", "Feedback"];
  function setStep(n) { // n = 1..4 is the current step; 5 means every step is done
    document.querySelectorAll("#stepper li").forEach((li) => {
      const s = Number(li.dataset.step);
      li.classList.toggle("is-current", s === n);
      li.classList.toggle("is-done", s < n);
      li.querySelector(".dot").textContent = s < n ? "✓" : String(s);
      if (s === n) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
    document.querySelectorAll("#step-bars span").forEach((bar, i) => {
      bar.className = i + 1 < n ? "is-done" : i + 1 === n ? "is-current" : "";
    });
    $("step-label").textContent = n > STEP_NAMES.length ? "Complete" : STEP_NAMES[n - 1];
    $("step-count").textContent = n > STEP_NAMES.length ? "All steps done" : `Step ${n} of ${STEP_NAMES.length}`;
  }

  async function api(path, { method = "GET", body, timeoutMs = 120000 } = {}) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const res = await fetch(API + path, {
        method,
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
        signal: ctl.signal,
        cache: "no-store",
        credentials: "same-origin",
      });
      let data = null;
      try { data = await res.json(); } catch { /* body was not JSON */ }
      if (!res.ok) {
        const msg = res.status === 429 ? "Too many requests. Please wait a minute and try again."
          : res.status === 422 ? "That input was not accepted."
          : res.status === 503 ? "The AI assistant is not available right now."
          : (data && typeof data.detail === "string" && data.detail)
          || (data && typeof data.error === "string" && data.error)
          || "Something went wrong. Please try again.";
        throw new ApiError(msg, res.status);
      }
      return data;
    } catch (err) {
      if (err.name === "AbortError") throw new ApiError("That took too long. It may still finish: use Resume with the diagnosis id.", 0);
      throw err;
    } finally {
      clearTimeout(t);
    }
  }

  // inline = true: small loader inside the question card only (follow-up questions).
  // inline = false: full banner that locks the page (first load after picking a code, resume, saving feedback).
  async function run(busyText, fn, busyTitle, inline = false) {
    hideError();
    if (inline) setInlineBusy(true, busyText); else setBusy(busyText, busyTitle);
    try {
      await fn();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) { await boot(); }
      else showError(err.message || "Something went wrong. Please try again.");
    } finally {
      if (inline) setInlineBusy(false); else clearBusy();
    }
  }

  // ---------- boot: who is allowed in? ----------
  async function boot() {
    try {
      const s = await api("/session");
      if (s.authenticated) { await init(); return; }
      screen(s.passcode_required ? "login" : "blocked");
    } catch (err) {
      showError(err.message);
    }
  }

  $("login-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const code = $("passcode").value;
    $("passcode").value = "";
    run("Checking…", async () => {
      await api("/session", { method: "POST", body: { passcode: code } });
      await init();
    });
  });

  async function init() {
    const models = await api("/catalog");
    const entry = models.models.find((m) => m.model_line === DEFAULT_MODEL);
    $("f-cases").textContent = String(entry ? entry.cases : 0);
    screen("start");
    setStep(1);
    try {
      const lastId = localStorage.getItem(LAST_ID_KEY);
      if (lastId) $("resume-id").value = lastId;
    } catch { /* storage unavailable (private mode, etc.) - Resume still works by pasting an id */ }
  }

  $("query-text").addEventListener("input", updateStartNote);

  // ---------- start / resume ----------
  $("start-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const queryText = $("query-text").value.trim();
    if (!queryText) { showError("Describe the problem or enter a fault code."); return; }
    begin(queryText, DEFAULT_MODEL);
  });
  $("resume-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const id = $("resume-id").value.trim();
    if (!id) { showError("Enter a diagnosis id to resume."); return; }
    resumeDiagnosis(id);
  });

  let currentTree = null; // {root, nodes[]} - the whole tree, walked locally
  let currentNodeId = null;
  let maxQuestions = 7;

  function findNode(id) { return (currentTree.nodes || []).find((n) => n.id === id); }
  function statusFor(tree, nodeId) {
    if (!tree) return "no_evidence";
    const node = findNode(nodeId);
    return node && node.conclusion ? "concluded" : "asking";
  }

  // Saved under the diagnosis id so "Resume" can restore the exact question reached, not just the tree -
  // the server only tracks the tree itself (see get_diagnosis), not which answers were given.
  function saveProgress() {
    if (!diagnosisId || !currentTree) return;
    try {
      localStorage.setItem(PROGRESS_PREFIX + diagnosisId, JSON.stringify({
        tree: currentTree, currentNodeId, history, future, vehicle, maxQuestions, evidence: lastEvidence,
      }));
      localStorage.setItem(LAST_ID_KEY, diagnosisId);
    } catch { /* storage unavailable (private mode, quota) - Resume falls back to the server copy */ }
  }
  function loadProgress(id) {
    try {
      const raw = localStorage.getItem(PROGRESS_PREFIX + id);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function begin(queryText, model) {
    run(`Checking your description for ${friendly(model)} and preparing the first question.`, async () => {
      const body = { model_line: model, query_text: queryText };
      const d = await api("/diagnoses", { method: "POST", body, timeoutMs: 600000 });
      history = [];
      future = [];
      lastConclusion = null;
      diagnosisId = d.diagnosis_id;
      vehicle = { dtc: d.dtc_code || "", model, query: queryText };
      currentTree = d.tree;
      currentNodeId = d.tree ? d.tree.root : null;
      maxQuestions = d.max_questions;
      enterWork(maxQuestions, d.evidence);
    }, "Preparing your diagnosis");
  }

  // Resume: same-browser diagnoses restore the exact question from local storage, with no server call at
  // all. A diagnosis id from another browser/device (or after local storage was cleared) falls back to the
  // server's cached tree and restarts at its root, since only the browser ever knew which answers were given.
  function resumeDiagnosis(idRaw) {
    const id = idRaw.trim().toLowerCase();
    if (!UUID_RE.test(id)) { showError("That doesn't look like a diagnosis id."); return; }
    hideError();
    const saved = loadProgress(id);
    if (saved && saved.tree) {
      history = saved.history || [];
      future = saved.future || [];
      lastConclusion = null;
      diagnosisId = id;
      vehicle = saved.vehicle || { dtc: "", model: DEFAULT_MODEL, query: "" };
      currentTree = saved.tree;
      currentNodeId = saved.currentNodeId;
      maxQuestions = saved.maxQuestions || 1;
      enterWork(maxQuestions, { ...(saved.evidence || {}), resumed: true });
      return;
    }
    run("Looking up that diagnosis…", async () => {
      const d = await api(`/diagnoses/${id}`);
      history = [];
      future = [];
      lastConclusion = null;
      diagnosisId = d.diagnosis_id;
      vehicle = { dtc: d.dtc_code || "", model: d.model_line, query: "" };
      currentTree = d.tree;
      currentNodeId = d.tree ? d.tree.root : null;
      maxQuestions = d.max_questions;
      enterWork(maxQuestions, { ...d.evidence, resumed: true });
    }, "Looking up that diagnosis");
  }

  const vehicleTitle = () => vehicle.dtc || "Symptom";
  // The prompt requires an ending that couldn't separate the causes to be low confidence with no part listed.
  const isInconclusive = (k) => k.confidence === "low" && k.parts.length === 0;

  function enterWork(maxQ, evidence) {
    screen("work");
    lastEvidence = evidence;
    $("v-code").textContent = vehicleTitle();
    const short = vehicle.query && vehicle.query.length > 90 ? `${vehicle.query.slice(0, 90)}…` : vehicle.query;
    $("v-model").textContent = (vehicle.dtc || !short) ? friendly(vehicle.model) : `${friendly(vehicle.model)} · “${short}”`;
    renderEvidence(evidence);
    renderState(statusFor(currentTree, currentNodeId), maxQ);
  }

  // ---------- answering: pure local tree walk, no server call ----------
  function submitAnswer(answer) {
    if (!currentTree || busy) return;
    const node = findNode(currentNodeId);
    history.push({ fromId: currentNodeId, question: node.question, answer, showed: "" });
    future = []; // a fresh answer replaces whatever branch "Forward" used to lead to
    const nextId = { yes: node.yes, no: node.no, not_sure: node.not_sure }[answer] || node.yes || node.no;
    currentNodeId = nextId;
    const next = findNode(nextId);
    renderState(next && next.conclusion ? "concluded" : "asking", maxQuestions);
  }
  // Back/Forward walk the same question/answer stack used for "Answered so far" - no server call, no re-ask.
  function goBack() {
    if (!currentTree || busy || history.length === 0) return;
    const entry = history.pop();
    future.push(entry);
    currentNodeId = entry.fromId;
    renderState("asking", maxQuestions);
  }
  function goForward() {
    if (!currentTree || busy || future.length === 0) return;
    const entry = future.pop();
    history.push(entry);
    const node = findNode(entry.fromId);
    const nextId = { yes: node.yes, no: node.no, not_sure: node.not_sure }[entry.answer] || node.yes || node.no;
    currentNodeId = nextId;
    const next = findNode(nextId);
    renderState(next && next.conclusion ? "concluded" : "asking", maxQuestions);
  }
  $("answers").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-answer]");
    if (btn) submitAnswer(btn.dataset.answer);
  });
  $("back-btn").addEventListener("click", goBack);
  $("forward-btn").addEventListener("click", goForward);
  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey || busy) return;
    if (["INPUT", "SELECT", "TEXTAREA", "SUMMARY"].includes(document.activeElement && document.activeElement.tagName)) return;
    if (ev.key === "ArrowLeft" && !$("back-btn").hidden) { ev.preventDefault(); goBack(); return; }
    if (ev.key === "ArrowRight" && !$("forward-btn").hidden) { ev.preventDefault(); goForward(); return; }
    if ($("answers").hidden) return;
    const map = { y: "yes", n: "no", s: "not_sure" };
    const answer = map[ev.key.toLowerCase()];
    if (answer) { ev.preventDefault(); submitAnswer(answer); }
  });

  // ---------- rendering (reads currentTree/currentNodeId, no server response needed) ----------
  function renderState(status, maxQuestions) {
    const node = currentTree ? findNode(currentNodeId) : null;
    asking = status === "asking";
    const concluded = status === "concluded";
    const conclusion = node && node.conclusion ? node.conclusion : null;
    const questionsAsked = history.length;
    setStep(concluded ? 3 : 2);
    renderHistory();
    $("question").textContent = asking ? (node && node.question)
      : concluded ? "Diagnosis complete. The result is below."
      : "There are no past repairs or manual text for this code and model, so there is nothing to base a diagnosis on.";
    $("q-count").textContent = asking ? `${questionsAsked + 1} of up to ${maxQuestions}` : "";
    $("q-meter").style.width = `${Math.min(100, Math.round(((asking ? questionsAsked : maxQuestions) / maxQuestions) * 100))}%`;
    show($("answers"), asking && !inlineBusy);
    show($("question-card"), !concluded);
    show($("conclusion-card"), Boolean(conclusion));
    const note = $("action-note");
    note.textContent = asking ? `Question ${questionsAsked + 1} of up to ${maxQuestions}. Answer Yes, No or Not sure.`
      : concluded ? (conclusion && isInconclusive(conclusion)
        ? "Not yet identified. Run the checks below, then give feedback on what you found."
        : "Diagnosis ready. Review the repair steps, then give feedback after the repair.")
      : "No diagnosis is possible for this code and model.";
    note.classList.toggle("strong", concluded);
    show($("copy-btn"), concluded);
    show($("print-btn"), concluded);
    show($("outcome-card"), concluded);
    if (conclusion) renderConclusion(conclusion);
    updateNavButtons();
    saveProgress();
    $((conclusion ? "c-title" : "q-title")).focus({ preventScroll: false });
  }

  function renderHistory() {
    const ol = $("history");
    clear(ol);
    show($("history-empty"), history.length === 0);
    show(ol, history.length > 0);
    history.forEach((h, i) => {
      ol.append(el("li", { cls: "row-item" }, [
        el("div", { cls: "h-top" }, [
          el("div", {}, [el("p", { cls: "h-n", text: `Q${i + 1}` }), el("p", { cls: "h-q", text: h.question })]),
          el("span", { cls: `tag ${h.answer}`, text: ANSWER_LABEL[h.answer] || h.answer }),
        ]),
        ...(h.showed ? [el("p", { cls: "h-note", text: `What it showed: ${h.showed}` })] : []),
      ]));
    });
  }

  function renderEvidence(ev) {
    const dl = $("evidence");
    clear(dl);
    if (!ev) return;
    const row = (k, v) => dl.append(el("dt", { text: k }), el("dd", { text: v }));
    const m = ev.manual || {};
    const manualLoaded = m.status === "used";
    if (ev.total_cases !== undefined) {
      row("Past repairs", ev.distinct_cases ? `${ev.total_cases} (${ev.distinct_cases} distinct)` : `${ev.total_cases}`);
    }
    if (ev.scope === "dtc_all_models") {
      row("Note", "No repairs for this code on this model, so repairs on other models are used.");
    } else if (ev.scope === "none") {
      row("Past repairs", "none found");
    }
    row("Service manual", manualLoaded ? `loaded${m.modules && m.modules.length ? `, ${m.modules.length} section(s)${m.truncated ? ", trimmed to fit" : ""}` : ""}` : "not available for this code");
    (ev.top_parts || []).slice(0, 3).forEach((p) => row("Often replaced", `${p.part} (${p.pct}%)`));
    if (ev.resumed) row("Note", "Resumed diagnosis.");
  }

  // Issue -> why -> fix, nothing hidden behind a click: a mechanic reading this once should have everything
  // needed, not a page of collapsed accordions and decorative counters.
  function renderConclusion(k) {
    lastConclusion = k;
    const box = $("conclusion");
    clear(box);
    const open = isInconclusive(k);
    const badge = $("confidence");
    badge.textContent = open ? "inconclusive" : `${k.confidence} confidence`;
    badge.className = `badge ${k.confidence}`;

    const list = (items) => el("ul", {}, items.map((t) => el("li", { text: t })));
    const card = (title, body) => el("section", { cls: "mini-card" }, [el("div", { cls: "mini-head" }, [el("h3", { text: title })]), body]);

    const why = [...k.evidence, ...(k.ruled_out || []).map((r) => `Ruled out: ${r}`), ...(k.uncertain || []).map((u) => `Still uncertain: ${u}`)];

    box.append(
      el("div", { cls: "problem-card" }, [
        el("p", { cls: "problem-kicker", text: open ? "Not yet identified" : "Most likely problem" }),
        el("p", { cls: "problem", text: k.problem }),
      ]),
      ...(k.confidence_note ? [el("p", { cls: "hint", text: `Confidence: ${k.confidence_note}` })] : []),
      el("div", { cls: "safety-banner", attrs: { role: "note" } }, [el("strong", { text: "Safety first. " }), document.createTextNode(k.safety_note)]),
      card("Why", list(why)),
      card(open ? "Checks to do" : "What to do", el("ol", { cls: "steps" }, k.fix_steps.map((t) => el("li", { text: t })))),
      ...(!open ? [card("Parts", k.parts.length
        ? el("div", { cls: "parts" }, k.parts.map((p) => el("span", { cls: "part", text: p })))
        : el("p", { cls: "muted", text: "No part replacement is supported by the evidence." }))] : []),
      card(open ? "How to confirm" : "Confirm the repair", list(k.verify_fix)),
      ...(k.alternatives.length ? [card("If it doesn't fix it", list(k.alternatives.map((a) => `${a.cause}: ${a.when_to_consider}`)))] : []),
    );
  }

  // ---------- copy / print ----------
  function summaryText() {
    const k = lastConclusion;
    if (!k) return "";
    const open = isInconclusive(k);
    const why = [...k.evidence, ...(k.ruled_out || []).map((r) => `Ruled out: ${r}`), ...(k.uncertain || []).map((u) => `Still uncertain: ${u}`)];
    const lines = [`${vehicleTitle()} on ${friendly(vehicle.model)}${vehicle.dtc ? "" : `: "${vehicle.query}"`}`, "", `${open ? "Not yet identified" : "Most likely problem"}: ${k.problem}`, `Confidence: ${k.confidence}${k.confidence_note ? ` (${k.confidence_note})` : ""}`,
      "", "Why:", ...why.map((w) => `- ${w}`),
      "", `${open ? "Checks to do" : "What to do"}:`, ...k.fix_steps.map((s, i) => `${i + 1}. ${s}`), "", `Parts: ${k.parts.length ? k.parts.join("; ") : "none supported"}`,
      "", `${open ? "How to confirm" : "Confirm the repair"}:`, ...k.verify_fix.map((v) => `- ${v}`), "", `Safety: ${k.safety_note}`];
    if (k.alternatives.length) lines.push("", "If it doesn't fix it:", ...k.alternatives.map((a) => `- ${a.cause}: ${a.when_to_consider}`));
    return lines.join("\n");
  }
  $("copy-btn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(summaryText());
      show($("copy-note"));
      setTimeout(() => show($("copy-note"), false), 2000);
    } catch {
      showError("Copying is not available in this browser. Use Print instead.");
    }
  });
  $("print-btn").addEventListener("click", () => window.print());

  // ---------- outcome feedback ----------
  // Final, read-only page shown once the feedback is saved.
  function showDone(o) {
    $("d-code").textContent = vehicle.dtc || "Not given (symptom only)";
    $("d-model").textContent = friendly(vehicle.model);
    $("d-result").textContent = { worked: "Fixed it", partly: "Partly fixed", not_worked: "Did not fix it" }[o.result] || o.result;
    $("d-id").textContent = diagnosisId;
    screen("done");
    setStep(5);
    window.scrollTo({ top: 0 });
    $("done-title").focus();
  }

  function markOutcomeSaved(o) {
    const s = $("outcome-saved");
    s.textContent = `Saved: ${RESULT_LABEL[o.result] || o.result}. Thank you.`;
    show(s);
    setStep(5);
  }
  $("outcome-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const result = new FormData(ev.target).get("result");
    const parts = $("parts").value.split(",").map((p) => p.trim()).filter(Boolean).slice(0, 10);
    run("Saving your feedback…", async () => {
      const o = await api(`/diagnoses/${diagnosisId}/outcome`, {
        method: "POST",
        body: { result, actual_cause: $("actual-cause").value.trim(), parts_replaced: parts, notes: $("outcome-notes").value.trim() },
      });
      markOutcomeSaved(o);
      showDone(o);
    });
  });

  // ---------- reset ----------
  function newDiagnosis() {
    diagnosisId = null;
    history = [];
    future = [];
    lastConclusion = null;
    hideError();
    show($("outcome-saved"), false);
    show($("copy-note"), false);
    $("outcome-form").reset();
    $("query-text").value = "";
    screen("start");
    setStep(1);
    window.scrollTo({ top: 0 });
  }
  $("reset").addEventListener("click", newDiagnosis);
  $("done-new").addEventListener("click", newDiagnosis);
  $("done-view").addEventListener("click", () => { screen("work"); setStep(5); window.scrollTo({ top: 0 }); });

  boot();
})();
