const STEP_LABELS = ["SITUATION", "OPPONENT", "GOAL", "DRAFT", "CRITIQUE", "FINAL"];

/** Parse the agent's raw think() output into labeled {label, cite, text} steps.
 *  Small/fast models often skip the SITUATION/.../FINAL scaffold entirely for
 *  simple factual questions and just answer directly — so this has to degrade
 *  gracefully: structured steps when the model followed the template, a plain
 *  paragraph when it free-wrote instead, and nothing at all when the "trace"
 *  is just the final answer restated (nothing extra to reveal). */
function parseTrace(raw, finalText) {
  const rawTrim = (raw || "").trim();
  const finalTrim = (finalText || "").trim();
  if (!rawTrim || rawTrim === finalTrim) return { kind: "none" };

  const lines = rawTrim.split("\n");
  const labelRe = new RegExp(`^\\*{0,2}(${STEP_LABELS.join("|")})\\*{0,2}\\s*(\\[[^\\]]*\\])?\\s*:?\\s*(.*)$`, "i");

  const steps = [];
  let current = null;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    const m = line.match(labelRe);
    if (m) {
      if (current) steps.push(current);
      current = { label: m[1].toUpperCase(), cite: (m[2] || "").replace(/^\[|\]$/g, ""), text: m[3] || "" };
    } else if (current && line) {
      current.text += (current.text ? " " : "") + line;
    }
  }
  if (current) steps.push(current);

  return steps.length ? { kind: "steps", steps } : { kind: "freeform", text: rawTrim };
}

function renderTrace(container, parsed) {
  container.innerHTML = "";
  if (parsed.kind === "steps") {
    parsed.steps.forEach((step, i) => {
      const el = document.createElement("div");
      el.className = "trace-step";
      el.style.animationDelay = `${i * 90}ms`;
      el.innerHTML = `
        <span class="step-label">${step.label}</span>
        ${step.cite ? `<span class="step-cite"> · ${escapeHtml(step.cite)}</span>` : ""}
        <div class="step-text">${escapeHtml(step.text)}</div>
      `;
      container.appendChild(el);
    });
  } else if (parsed.kind === "freeform") {
    const el = document.createElement("div");
    el.className = "trace-step trace-freeform";
    el.textContent = parsed.text;
    container.appendChild(el);
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function setState(card, state) {
  card.dataset.state = state;
  const pill = card.querySelector('[data-role="state-pill"]');
  pill.textContent = { idle: "idle", thinking: "thinking…", done: "settled", error: "error" }[state] || state;
}

async function runFighter(card, topic) {
  const id = card.dataset.fighter;
  const body = card.querySelector('[data-role="body"]');
  setState(card, "thinking");
  body.innerHTML = `<p class="idle-msg">Consulting the model…</p>`;

  try {
    const res = await fetch(`/api/duel/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);

    const qTrace = parseTrace(data.question_trace, data.question);
    const aTrace = parseTrace(data.answer_trace, data.answer);

    const traceBlock = (id, label, parsed) =>
      parsed.kind === "none"
        ? ""
        : `<details class="trace">
             <summary>${label}</summary>
             <div class="trace-steps" data-role="${id}"></div>
           </details>`;

    body.innerHTML = `
      <div class="qa-block">
        <span class="qa-label">Asks</span>
        <p class="qa-text">${escapeHtml(data.question)}</p>
      </div>
      ${traceBlock("q-trace", "Reasoning — the question", qTrace)}
      <div class="qa-block">
        <span class="qa-label">Answers itself</span>
        <p class="qa-text answer">${escapeHtml(data.answer)}</p>
      </div>
      ${traceBlock("a-trace", "Reasoning — the answer", aTrace)}
    `;
    if (qTrace.kind !== "none") renderTrace(body.querySelector('[data-role="q-trace"]'), qTrace);
    if (aTrace.kind !== "none") renderTrace(body.querySelector('[data-role="a-trace"]'), aTrace);
    setState(card, "done");
  } catch (err) {
    body.innerHTML = `<p class="error-msg">Couldn't finish this round: ${escapeHtml(err.message)}</p>`;
    setState(card, "error");
  }
}

function runDuel(topic) {
  const btn = document.getElementById("fight-btn");
  btn.disabled = true;
  const cards = Array.from(document.querySelectorAll(".fighter"));
  Promise.allSettled(cards.map((card) => runFighter(card, topic))).finally(() => {
    btn.disabled = false;
  });
}

document.getElementById("duel-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const topic = document.getElementById("topic-input").value.trim();
  if (topic) runDuel(topic);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const topic = chip.dataset.topic;
    document.getElementById("topic-input").value = topic;
    runDuel(topic);
  });
});
