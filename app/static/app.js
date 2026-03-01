const el = (id) => document.getElementById(id);

const fields = {
  provider: el("provider"),
  llmBaseUrl: el("llm-base-url"),
  model: el("model"),
  systemPrompt: el("system-prompt"),
  temperature: el("temperature"),
  topP: el("top-p"),
  maxTokens: el("max-tokens"),
  numCtx: el("num-ctx"),
  repeatPenalty: el("repeat-penalty"),
  seed: el("seed"),
  status: el("status"),
  question: el("question"),
  response: el("response"),
  history: el("history"),
  diagnostics: el("diagnostics"),
  topicButtons: el("topic-buttons"),
  ragCollections: el("rag-collections"),
  useRag: el("use-rag"),
  numCtxField: el("num-ctx-field"),
  seedField: el("seed-field"),
};
let availableRagCollections = [];
const diagnosticsState = {
  health: { status: "unknown", checkedAt: null, error: null },
  topics: {
    loadedAt: null,
    interestsCount: 0,
    trainingTopicsCount: 0,
    allTopicsCount: 0,
    error: null,
  },
  rag: {
    useRag: true,
    availableCollections: [],
    enabledCollections: [],
    lastContextChunks: 0,
    lastContextCollections: [],
    lastAskedAt: null,
    lastError: null,
  },
};

function isoNow() {
  return new Date().toISOString();
}

function showStatus(msg) {
  fields.status.textContent = msg;
}

async function copyDiagnostics() {
  const enabledRagCollections = Array.from(
    fields.ragCollections.querySelectorAll("input[type='checkbox']:checked")
  ).map((input) => input.value);

  const payload = {
    copied_at: isoNow(),
    ui_settings: {
      provider: fields.provider.value,
      llm_base_url: fields.llmBaseUrl.value.trim(),
      model: fields.model.value.trim(),
      use_rag: fields.useRag.checked,
      enabled_rag_collections: enabledRagCollections,
    },
    diagnostics: diagnosticsState,
    diagnostics_text: fields.diagnostics.textContent,
  };

  const text = JSON.stringify(payload, null, 2);
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(text);
    showStatus("Diagnostics copied to clipboard.");
    return;
  }

  // Fallback for environments where Clipboard API is unavailable.
  const temp = document.createElement("textarea");
  temp.value = text;
  document.body.appendChild(temp);
  temp.select();
  document.execCommand("copy");
  document.body.removeChild(temp);
  showStatus("Diagnostics copied to clipboard (fallback mode).");
}

function renderDiagnostics() {
  const healthLine = [
    "[health]",
    `status=${diagnosticsState.health.status}`,
    `checked_at=${diagnosticsState.health.checkedAt || "n/a"}`,
    diagnosticsState.health.error ? `error=${diagnosticsState.health.error}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  const topicsLine = [
    "[topics]",
    `loaded_at=${diagnosticsState.topics.loadedAt || "n/a"}`,
    `interests=${diagnosticsState.topics.interestsCount}`,
    `training_topics=${diagnosticsState.topics.trainingTopicsCount}`,
    `all_topics=${diagnosticsState.topics.allTopicsCount}`,
    diagnosticsState.topics.error ? `error=${diagnosticsState.topics.error}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  const ragLine = [
    "[rag]",
    `use_rag=${diagnosticsState.rag.useRag}`,
    `available=${diagnosticsState.rag.availableCollections.length}`,
    `enabled=${diagnosticsState.rag.enabledCollections.length}`,
    `last_chunks=${diagnosticsState.rag.lastContextChunks}`,
    `last_collections=${diagnosticsState.rag.lastContextCollections.join(",") || "none"}`,
    `last_asked_at=${diagnosticsState.rag.lastAskedAt || "n/a"}`,
    diagnosticsState.rag.lastError ? `error=${diagnosticsState.rag.lastError}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  fields.diagnostics.textContent = [healthLine, topicsLine, ragLine].join("\n");
}

function currentSettingsPayload() {
  const seedValue = fields.seed.value.trim();
  const enabledRagCollections = Array.from(
    fields.ragCollections.querySelectorAll("input[type='checkbox']:checked")
  ).map((input) => input.value);
  diagnosticsState.rag.enabledCollections = enabledRagCollections;
  diagnosticsState.rag.useRag = fields.useRag.checked;
  renderDiagnostics();
  return {
    provider: fields.provider.value,
    llm_base_url: fields.llmBaseUrl.value.trim(),
    model: fields.model.value.trim(),
    system_prompt: fields.systemPrompt.value,
    enabled_rag_collections: enabledRagCollections,
    tweaks: {
      temperature: Number(fields.temperature.value),
      top_p: Number(fields.topP.value),
      max_tokens: Number(fields.maxTokens.value),
      num_ctx: Number(fields.numCtx.value),
      repeat_penalty: Number(fields.repeatPenalty.value),
      seed: seedValue ? Number(seedValue) : null,
    },
  };
}

function applySettings(settings) {
  fields.provider.value = settings.provider;
  fields.llmBaseUrl.value = settings.llm_base_url;
  fields.model.value = settings.model;
  fields.systemPrompt.value = settings.system_prompt;
  fields.temperature.value = settings.tweaks.temperature;
  fields.topP.value = settings.tweaks.top_p;
  fields.maxTokens.value = settings.tweaks.max_tokens;
  fields.numCtx.value = settings.tweaks.num_ctx;
  fields.repeatPenalty.value = settings.tweaks.repeat_penalty;
  fields.seed.value = settings.tweaks.seed ?? "";
  renderRagCollectionCheckboxes(
    availableRagCollections,
    settings.enabled_rag_collections || []
  );
  toggleProviderFields();
}

function toggleProviderFields() {
  const provider = fields.provider.value;
  fields.numCtxField.style.display = provider === "ollama" ? "grid" : "none";
  fields.seedField.style.display = provider === "llama_cpp" ? "grid" : "none";
}

async function loadSettings() {
  const response = await fetch("/api/settings");
  const data = await response.json();
  applySettings(data);
  diagnosticsState.rag.enabledCollections = data.enabled_rag_collections || [];
  diagnosticsState.rag.useRag = fields.useRag.checked;
  renderDiagnostics();
}

function renderRagCollectionCheckboxes(availableCollections, enabledCollections) {
  fields.ragCollections.innerHTML = "";
  if (!availableCollections.length) {
    fields.ragCollections.textContent =
      "No Qdrant collections configured. Set RAG_COLLECTIONS or RAG_COLLECTION_1..3.";
    return;
  }

  const enabledSet = new Set(enabledCollections);
  diagnosticsState.rag.availableCollections = availableCollections;
  diagnosticsState.rag.enabledCollections = enabledCollections;
  availableCollections.forEach((collectionName) => {
    const wrapper = document.createElement("label");
    wrapper.className = "muted";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = collectionName;
    checkbox.checked = enabledSet.has(collectionName);
    checkbox.addEventListener("change", () => {
      diagnosticsState.rag.enabledCollections = Array.from(
        fields.ragCollections.querySelectorAll("input[type='checkbox']:checked")
      ).map((input) => input.value);
      renderDiagnostics();
    });
    wrapper.appendChild(checkbox);
    wrapper.append(` ${collectionName}`);
    fields.ragCollections.appendChild(wrapper);
  });
  renderDiagnostics();
}

async function loadRagCollections() {
  const response = await fetch("/api/rag/collections");
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  availableRagCollections = data.available_collections || [];
  renderRagCollectionCheckboxes(
    availableRagCollections,
    data.enabled_collections || []
  );
  renderDiagnostics();
}

async function saveSettings() {
  const response = await fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentSettingsPayload()),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Save failed: ${body}`);
  }
  const data = await response.json();
  applySettings(data);
}

async function loadTopics() {
  fields.topicButtons.innerHTML = "";
  try {
    const response = await fetch("/api/topics");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const data = await response.json();
    const topics = data.all_topics || [];
    topics.forEach((topic) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = topic;
      button.addEventListener("click", async () => {
        fields.question.value = topic;
        await askQuestion();
      });
      fields.topicButtons.appendChild(button);
    });
    diagnosticsState.topics.loadedAt = isoNow();
    diagnosticsState.topics.interestsCount = (data.interests || []).length;
    diagnosticsState.topics.trainingTopicsCount = (data.training_topics || []).length;
    diagnosticsState.topics.allTopicsCount = topics.length;
    diagnosticsState.topics.error = null;
    renderDiagnostics();
    showStatus(
      `Loaded interests=${(data.interests || []).length}, training_topics=${(data.training_topics || []).length}.`
    );
  } catch (err) {
    diagnosticsState.topics.error = err instanceof Error ? err.message : String(err);
    renderDiagnostics();
    showStatus(`Topic load failed: ${diagnosticsState.topics.error}`);
  }
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=30");
  if (!response.ok) {
    fields.history.textContent = "Unable to load history.";
    return;
  }
  const rows = await response.json();
  fields.history.textContent = rows
    .map(
      (item) =>
        `[${item.created_at}] (${item.provider}/${item.model})\nQ: ${item.question}\nA: ${item.answer}\n---`
    )
    .join("\n");
}

async function askQuestion() {
  const question = fields.question.value.trim();
  if (!question) {
    showStatus("Enter a question first.");
    return;
  }
  showStatus("Saving settings and contacting model...");
  fields.response.textContent = "";
  await saveSettings();

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      use_rag: fields.useRag.checked,
    }),
  });
  if (!response.ok) {
    const errBody = await response.text();
    fields.response.textContent = `Request failed: ${errBody}`;
    diagnosticsState.rag.lastError = errBody;
    diagnosticsState.rag.lastAskedAt = isoNow();
    renderDiagnostics();
    showStatus("Chat request failed.");
    return;
  }
  const data = await response.json();
  fields.response.textContent = data.answer;
  diagnosticsState.rag.lastContextChunks = (data.context || []).length;
  diagnosticsState.rag.lastContextCollections = Array.from(
    new Set((data.context || []).map((chunk) => chunk.collection))
  );
  diagnosticsState.rag.lastError = null;
  diagnosticsState.rag.lastAskedAt = isoNow();
  renderDiagnostics();
  showStatus(`Done. Context chunks used: ${data.context.length}`);
  await loadHistory();
}

el("save-settings").addEventListener("click", async () => {
  try {
    await saveSettings();
    showStatus("Settings saved.");
  } catch (err) {
    showStatus(err instanceof Error ? err.message : String(err));
  }
});

el("ask").addEventListener("click", async () => {
  try {
    await askQuestion();
  } catch (err) {
    showStatus(err instanceof Error ? err.message : String(err));
  }
});

el("refresh-topics").addEventListener("click", async () => {
  showStatus("Refreshing topics...");
  await loadTopics();
  showStatus("Topics refreshed.");
});

el("copy-diagnostics").addEventListener("click", async () => {
  try {
    await copyDiagnostics();
  } catch (err) {
    showStatus(`Copy failed: ${err instanceof Error ? err.message : String(err)}`);
  }
});

fields.provider.addEventListener("change", toggleProviderFields);
fields.useRag.addEventListener("change", () => {
  diagnosticsState.rag.useRag = fields.useRag.checked;
  renderDiagnostics();
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    diagnosticsState.health.status = "ok";
    diagnosticsState.health.error = null;
  } catch (err) {
    diagnosticsState.health.status = "error";
    diagnosticsState.health.error = err instanceof Error ? err.message : String(err);
  }
  diagnosticsState.health.checkedAt = isoNow();
  renderDiagnostics();
}

async function bootstrap() {
  renderDiagnostics();
  await checkHealth();
  await loadRagCollections();
  await loadSettings();
  await loadTopics();
  await loadHistory();
  showStatus("Ready.");
}

bootstrap().catch((err) => {
  showStatus(`Startup failed: ${err.message}`);
});
