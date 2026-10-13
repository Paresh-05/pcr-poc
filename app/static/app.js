"use strict";

(() => {
  const API = "/ui-api"; // the server holds the API key; this page never sees one
  const $ = (id) => document.getElementById(id);
  const DTC_RE = /^[A-Z][0-9A-Z]{5,7}$/;
  const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  const ANSWER_LABEL = { yes: "Yes", no: "No", not_sure: "Not sure" };
  const RESULT_LABEL = { worked: "fixed", partly: "partly fixed", not_worked: "did not fix" };
  const UPPER_WORDS = new Set(["MG", "EV", "BS4", "BS6", "SUV"]);

  let diagnosisId = null;
  let history = []; // [{question, answer, note}]
  let dtcOptions = new Map(); // dtc_code -> {cases, has_manual} for the selected model
  let lastConclusion = null;
  let vehicle = { dtc: "", model: "" };
  let timer = null;
  let busy = false;

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
    $("status-time").textContent = "0 s";
    show($("status"));
    clearInterval(timer);
    timer = setInterval(() => { $("status-time").textContent = `${Math.round((Date.now() - started) / 1000)} s`; }, 1000);
    document.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  }
  function clearBusy() {
    busy = false;
    clearInterval(timer);
    show($("status"), false);
    document.querySelectorAll("button").forEach((b) => { b.disabled = false; });
  }
  function showError(msg) { const e = $("error"); e.textContent = msg; show(e); }
  const hideError = () => show($("error"), false);

  function screen(name) {
    for (const id of ["login", "blocked", "start", "work"]) show($(`screen-${id}`), id === name);
  }
  function setStep(n) {
    document.querySelectorAll("#stepper li").forEach((li) => {
      const s = Number(li.dataset.step);
      li.classList.toggle("is-current", s === n);
      li.classList.toggle("is-done", s < n);
    });
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

  async function run(busyText, fn, busyTitle) {
    hideError();
    setBusy(busyText, busyTitle);
    try {
      await fn();
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) { await boot(); }
      else showError(err.message || "Something went wrong. Please try again.");
    } finally {
      clearBusy();
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
    const sel = $("model");
    clear(sel);
    for (const m of models.models) sel.append(el("option", { text: friendly(m.model_line), attrs: { value: m.model_line } }));
    screen("start");
    setStep(1);
    await loadDtcs();
  }

  // ---------- pickers ----------
  async function loadDtcs() {
    const model = $("model").value;
    if (!model) return;
    const data = await api(`/catalog/${encodeURIComponent(model)}`);
    dtcOptions = new Map(data.dtcs.map((d) => [d.dtc_code, d]));
    const select = $("dtc");
    clear(select);
    select.append(el("option", { text: "Select a fault code", attrs: { value: "" } }));
    for (const d of data.dtcs) select.append(el("option", { text: d.dtc_code, attrs: { value: d.dtc_code } }));
    updateDtcMeta();
  }
  $("model").addEventListener("change", () => run("Loading fault codes…", loadDtcs));

  function updateDtcMeta() {
    const code = $("dtc").value;
    $("dtc-meta").textContent = !dtcOptions.size
      ? "No fault codes with complete manual procedures are available for this model."
      : !code ? "Only fault codes with complete manuals and matching past cases are shown." : "";
  }
  $("dtc").addEventListener("change", updateDtcMeta);

  // ---------- start / resume ----------
  $("start-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const dtc = $("dtc").value;
    if (!DTC_RE.test(dtc)) { showError("Enter a valid fault code, for example P187973."); return; }
    begin(dtc, $("model").value);
  });

  function begin(dtc, model) {
    run(`Checking ${dtc} for ${friendly(model)} and preparing the first question.`, async () => {
      const d = await api("/diagnoses", { method: "POST", body: { dtc_code: dtc, model_line: model }, timeoutMs: 600000 });
      history = [];
      lastConclusion = null;
      diagnosisId = d.diagnosis_id;
      vehicle = { dtc, model };
      enterWork(d, d.evidence);
      await refreshCost();
    }, "Preparing your diagnosis");
  }

  $("resume-btn").addEventListener("click", () => {
    const id = $("resume-id").value.trim();
    if (!UUID_RE.test(id)) { showError("That is not a diagnosis id."); return; }
    run("Loading the diagnosis…", async () => {
      const d = await api(`/diagnoses/${id}`);
      diagnosisId = d.diagnosis_id;
      history = d.steps.map((s) => ({ question: s.question, answer: s.answer, note: s.note || "" }));
      vehicle = { dtc: d.dtc_code, model: d.model_line };
      enterWork(d, { scope: d.scope, manual: { status: d.manual_status, modules: [] }, resumed: true });
      if (d.outcome) markOutcomeSaved(d.outcome);
      await refreshCost();
    });
  });

  function enterWork(d, evidence) {
    screen("work");
    $("v-code").textContent = vehicle.dtc;
    $("v-model").textContent = friendly(vehicle.model);
    renderEvidence(evidence);
    renderState(d);
  }

  // ---------- answering ----------
  function submitAnswer(answer) {
    if (!diagnosisId || busy) return;
    const note = $("note").value.trim();
    const asked = $("question").textContent;
    run("Updating the likely causes…", async () => {
      const d = await api(`/diagnoses/${diagnosisId}/answers`, { method: "POST", body: { answer, note } });
      history.push({ question: asked, answer, note });
      $("note").value = "";
      $("note-wrap").open = false;
      renderState(d);
      await refreshCost();
    });
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

  // ---------- rendering ----------
  function renderState(d) {
    const asking = d.status === "asking";
    const concluded = d.status === "concluded";
    setStep(concluded ? 3 : 2);
    renderCandidates(d.candidates || []);
    renderHistory();
    $("question").textContent = asking ? d.question
      : concluded ? "Diagnosis complete. The result is below."
      : "There are no past repairs or manual text for this code and model, so there is nothing to base a diagnosis on.";
    $("q-count").textContent = asking ? `${d.questions_asked + 1} of up to ${d.max_questions}` : "";
    $("q-meter").style.width = `${Math.min(100, Math.round(((asking ? d.questions_asked : d.max_questions) / d.max_questions) * 100))}%`;
    show($("answers"), asking);
    show($("note-wrap"), asking);
    show($("conclusion-card"), Boolean(d.conclusion));
    show($("outcome-card"), concluded);
    if (d.conclusion) renderConclusion(d.conclusion);
    $("diag-id").textContent = `Diagnosis id: ${d.diagnosis_id}`;
    $((d.conclusion ? "c-title" : "q-title")).focus({ preventScroll: false });
  }

  function renderCandidates(cands) {
    const ul = $("candidates");
    clear(ul);
    const fills = [];
    for (const c of cands) {
      const fill = el("span");
      fills.push([fill, Math.max(0, Math.min(100, c.likelihood))]);
      ul.append(el("li", {}, [
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
    history.forEach((h) => {
      ol.append(el("li", {}, [
        el("span", { text: h.question }),
        el("span", { cls: `tag ${h.answer}`, text: ANSWER_LABEL[h.answer] || h.answer }),
        ...(h.note ? [el("span", { cls: "muted", text: `Note: ${h.note}` })] : []),
      ]));
    });
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
      row("Past repairs", `${ev.total_cases}`);
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
    const list = (cls, items) => el(cls === "steps" ? "ol" : "ul", { cls: cls === "steps" ? "steps" : "" }, items.map((t) => el("li", { text: t })));
    box.append(
      el("p", { cls: "problem", text: k.problem }),
      el("h3", { text: "Why we think so" }), list("", k.evidence),
      el("h3", { text: "Repair steps" }), list("steps", k.fix_steps),
      el("h3", { text: "Parts" }),
      k.parts.length ? el("div", { cls: "parts" }, k.parts.map((p) => el("span", { cls: "part", text: p }))) : el("p", { cls: "muted", text: "No part replacement is supported by the evidence." }),
      el("h3", { text: "Confirm the repair" }), list("", k.verify_fix),
      el("h3", { text: "If it does not fix it" }),
      k.alternatives.length ? list("", k.alternatives.map((a) => `${a.cause}: ${a.when_to_consider}`)) : el("p", { cls: "muted", text: "No alternatives listed." }),
      el("p", { cls: "safety", text: k.safety_note }),
    );
  }

  async function refreshCost() {
    const data = await api(`/diagnoses/${diagnosisId}/cost`);
    const t = data.run_total;
    const dl = $("cost");
    const tbody = $("calls").querySelector("tbody");
    clear(dl);
    clear(tbody);
    if (!t) { dl.append(el("dd", { text: "No model calls yet." })); show($("calls"), false); return; }
    const row = (k, v) => dl.append(el("dt", { text: k }), el("dd", { text: v }));
    row("Cost of this run", `$${t.cost_usd.toFixed(5)}`);
    row("Model calls", String(t.calls));
    row("Input tokens", `${t.input_tokens} (${t.cached_input_tokens} cached)`);
    row("Output tokens", `${t.output_tokens} (${t.reasoning_tokens} thinking)`);
    for (const c of data.calls) {
      tbody.append(el("tr", {}, [c.purpose, c.input_tokens, c.cached_input_tokens, c.output_tokens, c.reasoning_tokens,
        `${(c.latency_ms / 1000).toFixed(1)}s`, c.cost_usd === null ? "n/a" : c.cost_usd.toFixed(5)]
        .map((v) => el("td", { text: String(v) }))));
    }
    show($("calls"));
  }

  // ---------- copy / print ----------
  function summaryText() {
    const k = lastConclusion;
    if (!k) return "";
    const lines = [`${vehicle.dtc} on ${friendly(vehicle.model)}`, "", `Most likely problem: ${k.problem}`, `Confidence: ${k.confidence}`, "", "Why:", ...k.evidence.map((e) => `- ${e}`),
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
  function markOutcomeSaved(o) {
    const s = $("outcome-saved");
    s.textContent = `Saved: ${RESULT_LABEL[o.result] || o.result}. Thank you.`;
    show(s);
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
    });
  });

  // ---------- reset ----------
  $("reset").addEventListener("click", () => {
    diagnosisId = null;
    history = [];
    lastConclusion = null;
    hideError();
    show($("outcome-saved"), false);
    $("outcome-form").reset();
    screen("start");
    setStep(1);
    window.scrollTo({ top: 0 });
  });

  boot();
})();
