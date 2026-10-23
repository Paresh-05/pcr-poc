"use strict";

(() => {
  const API = "/ui-api"; // the server holds the API key; this page never sees one
  const $ = (id) => document.getElementById(id);
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const ANSWER_LABEL = { yes: "Yes", no: "No", not_sure: "Not sure" };
  const RESULT_LABEL = { worked: "fixed", partly: "partly fixed", not_worked: "did not fix" };
  const UPPER_WORDS = new Set(["MG", "EV", "BS4", "BS6", "SUV"]);
  const DEFAULT_MODEL = "MG_COMET_EV"; // only model supported for now; no picker shown

  let diagnosisId = null;
  let history = []; // [{question, answer}]
  let lastConclusion = null;
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
  }

  $("query-text").addEventListener("input", updateStartNote);

  // ---------- start / resume ----------
  $("start-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const queryText = $("query-text").value.trim();
    if (!queryText) { showError("Describe the problem or enter a fault code."); return; }
    begin(queryText, DEFAULT_MODEL);
  });

  let currentTree = null; // {root, nodes[]} - the whole tree, walked locally
  let currentNodeId = null;

  function findNode(id) { return (currentTree.nodes || []).find((n) => n.id === id); }

  function begin(queryText, model) {
    run(`Checking your description for ${friendly(model)} and preparing the first question.`, async () => {
      const body = { model_line: model, query_text: queryText };
      const d = await api("/diagnoses", { method: "POST", body, timeoutMs: 600000 });
      history = [];
      lastConclusion = null;
      diagnosisId = d.diagnosis_id;
      vehicle = { dtc: d.dtc_code || "(from description)", model };
      currentTree = d.tree;
      currentNodeId = d.tree ? d.tree.root : null;
      enterWork(d, d.evidence);
      await refreshCost();
    }, "Preparing your diagnosis");
  }

  function enterWork(d, evidence) {
    screen("work");
    $("v-code").textContent = vehicle.dtc;
    $("v-model").textContent = friendly(vehicle.model);
    renderEvidence(evidence);
    renderState(d.status, d.max_questions);
  }

  // ---------- answering: pure local tree walk, no server call ----------
  function submitAnswer(answer) {
    if (!currentTree || busy) return;
    const node = findNode(currentNodeId);
    history.push({ question: node.question, answer, showed: "" });
    const nextId = { yes: node.yes, no: node.no, not_sure: node.not_sure }[answer] || node.yes || node.no;
    currentNodeId = nextId;
    const next = findNode(nextId);
    renderState(next && next.conclusion ? "concluded" : "asking", 5);
  }
  $("answers").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-answer]");
    if (btn) submitAnswer(btn.dataset.answer);
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey || busy || $("answers").hidden) return;
    if (["INPUT", "SELECT", "TEXTAREA", "SUMMARY"].includes(document.activeElement && document.activeElement.tagName)) return;
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
    renderCandidates((node && node.candidates) || []);
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
      : concluded ? "Diagnosis ready. Review the repair steps, then give feedback after the repair."
      : "No diagnosis is possible for this code and model.";
    note.classList.toggle("strong", concluded);
    show($("copy-btn"), concluded);
    show($("print-btn"), concluded);
    show($("outcome-card"), concluded);
    if (conclusion) renderConclusion(conclusion);
    $("diag-id").textContent = `Diagnosis id: ${diagnosisId}`;
    $((conclusion ? "c-title" : "q-title")).focus({ preventScroll: false });
  }

  function renderCandidates(cands) {
    const ul = $("candidates");
    clear(ul);
    const fills = [];
    for (const c of cands) {
      const fill = el("span");
      fills.push([fill, Math.max(0, Math.min(100, c.likelihood))]);
      ul.append(el("li", { cls: `rank-${Math.min(cands.indexOf(c) + 1, 3)}` }, [
        el("div", { cls: "cand-head" }, [el("span", { text: c.cause }), el("strong", { text: `${c.likelihood}%` })]),
        el("div", { cls: "bar", attrs: { role: "img", "aria-label": `${c.likelihood} percent` } }, [fill]),
        el("div", { cls: "cand-why", text: c.reasoning }),
      ]));
    }
    requestAnimationFrame(() => requestAnimationFrame(() => fills.forEach(([f, w]) => { f.style.width = `${w}%`; })));
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
    const count = (a) => history.filter((h) => h.answer === a).length;
    $("t-yes").textContent = String(count("yes"));
    $("t-no").textContent = String(count("no"));
    $("t-unsure").textContent = String(count("not_sure"));
  }

  function renderEvidence(ev) {
    const chips = $("v-evidence");
    const dl = $("evidence");
    clear(chips);
    clear(dl);
    if (!ev) return;
    const row = (k, v) => dl.append(el("dt", { text: k }), el("dd", { text: v }));
    const m = ev.manual || {};
    const manualLoaded = m.status === "used";
    if (ev.total_cases !== undefined) {
      chips.append(el("span", { cls: "chip", text: `${ev.total_cases} past repairs` }));
      row("Past repairs", ev.distinct_cases ? `${ev.total_cases} (${ev.distinct_cases} distinct)` : `${ev.total_cases}`);
    }
    if (ev.scope === "dtc_all_models") {
      chips.append(el("span", { cls: "chip warn", text: "From other models" }));
      row("Note", "No repairs for this code on this model, so repairs on other models are used.");
    } else if (ev.scope === "none") {
      row("Past repairs", "none found");
    }
    chips.append(el("span", { cls: `chip ${manualLoaded ? "good" : "warn"}`, text: manualLoaded ? "Service manual loaded" : "No manual text" }));
    row("Service manual", manualLoaded ? `loaded${m.modules && m.modules.length ? `, ${m.modules.length} section(s)${m.truncated ? ", trimmed to fit" : ""}` : ""}` : "not available for this code");
    (ev.top_parts || []).slice(0, 3).forEach((p) => row("Often replaced", `${p.part} (${p.pct}%)`));
    if (ev.resumed) row("Note", "Resumed diagnosis.");
  }

  function renderConclusion(k) {
    lastConclusion = k;
    const box = $("conclusion");
    clear(box);
    const badge = $("confidence");
    badge.textContent = `${k.confidence} confidence`;
    badge.className = `badge ${k.confidence}`;

    const list = (items, cls = "") => el(cls ? "ol" : "ul", cls ? { cls } : {}, items.map((t) => el("li", { text: t })));
    const stat = (n, label) => el("div", {}, [el("p", { cls: "tally-n", text: String(n) }), el("p", { cls: "tally-l", text: label })]);
    const meta = (text) => (text ? [el("span", { cls: "acc-meta", text })] : []);
    const card = (title, metaText, body) => el("section", { cls: "mini-card" }, [
      el("div", { cls: "mini-head" }, [el("h3", { text: title }), ...meta(metaText)]), body,
    ]);
    const fold = (title, body) => el("details", { cls: "acc" }, [
      el("summary", {}, [el("span", { cls: "acc-title", text: title })]),
      el("div", { cls: "acc-body" }, [body]),
    ]);

    box.append(
      el("div", { cls: "problem-card" }, [
        el("p", { cls: "problem-kicker", text: "Most likely problem" }),
        el("p", { cls: "problem", text: k.problem }),
        el("div", { cls: "problem-stats" }, [stat(k.fix_steps.length, "Repair steps"), stat(k.parts.length, "Parts"), stat(history.length, "Questions answered")]),
      ]),
      ...(k.confidence_note ? [el("p", { cls: "hint", text: `Confidence: ${k.confidence_note}` })] : []),
      el("div", { cls: "safety-banner", attrs: { role: "note" } }, [el("strong", { text: "Safety first. " }), document.createTextNode(k.safety_note)]),
      card("Repair steps", `${k.fix_steps.length} step${k.fix_steps.length === 1 ? "" : "s"}`, list(k.fix_steps, "steps")),
      el("div", { cls: "two-up" }, [
        card("Parts", k.parts.length ? String(k.parts.length) : "none",
          k.parts.length ? el("div", { cls: "parts" }, k.parts.map((p) => el("span", { cls: "part", text: p }))) : el("p", { cls: "muted", text: "No part replacement is supported by the evidence." })),
        card("Confirm the repair", "", list(k.verify_fix)),
      ]),
      ...((k.how_we_got_here || []).length ? [card("How we reached this finding", `${history.length} answer${history.length === 1 ? "" : "s"}`, list(k.how_we_got_here, "steps"))] : []),
      el("div", { cls: "acc-list" }, [
        fold("Why we think so", list(k.evidence)),
        ...((k.ruled_out || []).length ? [fold("What was ruled out", list(k.ruled_out))] : []),
        ...((k.uncertain || []).length ? [fold("What is still uncertain", list(k.uncertain))] : []),
        fold("If it does not fix it", k.alternatives.length ? list(k.alternatives.map((a) => `${a.cause}: ${a.when_to_consider}`)) : el("p", { cls: "muted", text: "No alternatives listed." })),
      ]),
    );
  }

  async function refreshCost() {
    const data = await api(`/diagnoses/${diagnosisId}/cost`);
    const t = data.run_total;
    const dl = $("cost");
    const list = $("calls");
    clear(dl);
    clear(list);
    if (!t) { dl.append(el("dd", { text: "No model calls yet." })); show(list, false); return; }
    const n = (v) => Number(v).toLocaleString("en-US");
    const row = (k, v) => dl.append(el("dt", { text: k }), el("dd", { text: v }));
    const pk = data.pack || {};
    row("Questions and result", `$${(pk.built_in_this_run && pk.cost_usd !== null ? t.cost_usd - pk.cost_usd : t.cost_usd).toFixed(5)}`);
    if (pk.cost_usd !== null && pk.cost_usd !== undefined) {
      row("Knowledge pack", `$${pk.cost_usd.toFixed(5)} (built once for this code${pk.built_in_this_run ? ", built during this run" : ", shared by every diagnosis"})`);
    }
    row("Total for this run", `$${t.cost_usd.toFixed(5)}${pk.built_in_this_run || pk.cost_usd === null || pk.cost_usd === undefined ? "" : ` (+ $${pk.cost_usd.toFixed(5)} pack, built earlier = $${(t.cost_usd + pk.cost_usd).toFixed(5)} all-in)`}`);
    row("Model calls", String(t.calls));
    row("Input tokens", `${n(t.input_tokens)} (${n(t.cached_input_tokens)} cached)`);
    row("Output tokens", `${n(t.output_tokens)} (${n(t.reasoning_tokens)} thinking)`);
    for (const c of data.calls) {
      list.append(el("li", {}, [
        el("div", { cls: "call-top" }, [
          el("strong", { text: c.purpose }),
          el("span", { cls: "muted", text: `${(c.latency_ms / 1000).toFixed(1)} s` }),
          el("span", { cls: "call-cost", text: c.cost_usd === null ? "n/a" : `$${c.cost_usd.toFixed(5)}` }),
        ]),
        el("div", { cls: "call-tokens", text: `in ${n(c.input_tokens)} (${n(c.cached_input_tokens)} cached) · out ${n(c.output_tokens)} (${n(c.reasoning_tokens)} thinking)` }),
      ]));
    }
    show(list);
  }

  // ---------- copy / print ----------
  function summaryText() {
    const k = lastConclusion;
    if (!k) return "";
    const lines = [`${vehicle.dtc} on ${friendly(vehicle.model)}`, "", `Most likely problem: ${k.problem}`, `Confidence: ${k.confidence}${k.confidence_note ? ` (${k.confidence_note})` : ""}`, "", "Why:", ...k.evidence.map((e) => `- ${e}`),
      "", "How we reached this finding:", ...(k.how_we_got_here || []).map((h) => `- ${h}`),
      "", "Repair steps:", ...k.fix_steps.map((s, i) => `${i + 1}. ${s}`), "", `Parts: ${k.parts.length ? k.parts.join("; ") : "none supported"}`,
      "", "Confirm the repair:", ...k.verify_fix.map((v) => `- ${v}`), "", `Safety: ${k.safety_note}`];
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
    $("d-code").textContent = vehicle.dtc;
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
