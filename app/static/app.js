const el = (id) => document.getElementById(id);

const fields = {
  provider: el("provider"),
  llmBaseUrl: el("llm-base-url"),
  modelSelect: el("model-select"),
  modelCustom: el("model-custom"),
  refreshModels: el("refresh-models"),
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
  metrics: el("metrics"),
  performanceChart: el("performance-chart"),
  history: el("history"),
  diagnostics: el("diagnostics"),
  footerMeta: el("footer-meta"),
  topicButtons: el("topic-buttons"),
  ragCollections: el("rag-collections"),
  useRag: el("use-rag"),
  temporaryChat: el("temporary-chat"),
  numCtxField: el("num-ctx-field"),
  seedField: el("seed-field"),
};
let availableRagCollections = [];
let availableModels = [];
const PERFORMANCE_HISTORY_LIMIT = 20;
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
    temporaryChat: false,
    availableCollections: [],
    enabledCollections: [],
    lastContextChunks: 0,
    lastContextCollections: [],
    lastAskedAt: null,
    lastError: null,
  },
  performance: {
    totalLatencyMs: null,
    ragLatencyMs: null,
    llmLatencyMs: null,
    promptTokens: null,
    completionTokens: null,
    totalTokens: null,
    tokensPerSecond: null,
  },
  performanceHistory: [],
};

function isoNow() {
  return new Date().toISOString();
}

function showStatus(msg) {
  if (fields.status) {
    fields.status.textContent = msg;
  }
}

function initHelpTips() {
  const tipButtons = Array.from(document.querySelectorAll(".help-tip"));
  const hideAll = () => {
    tipButtons.forEach((button) => {
      const tipId = button.dataset.tipId;
      if (!tipId) {
        return;
      }
      const tip = el(tipId);
      if (!tip) {
        return;
      }
      tip.hidden = true;
      button.setAttribute("aria-expanded", "false");
    });
  };

  tipButtons.forEach((button) => {
    const tipId = button.dataset.tipId;
    if (!tipId) {
      return;
    }
    const tip = el(tipId);
    if (!tip) {
      return;
    }
    button.setAttribute("aria-label", "Show help");
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const wasHidden = tip.hidden;
      hideAll();
      if (wasHidden) {
        tip.hidden = false;
        button.setAttribute("aria-expanded", "true");
      }
    });
  });

  document.addEventListener("click", hideAll);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideAll();
    }
  });
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
      model: getCurrentModel(),
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
  if (!fields.diagnostics) {
    return;
  }
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
    `temporary_chat=${diagnosticsState.rag.temporaryChat}`,
    `available=${diagnosticsState.rag.availableCollections.length}`,
    `enabled=${diagnosticsState.rag.enabledCollections.length}`,
    `last_chunks=${diagnosticsState.rag.lastContextChunks}`,
    `last_collections=${diagnosticsState.rag.lastContextCollections.join(",") || "none"}`,
    `last_asked_at=${diagnosticsState.rag.lastAskedAt || "n/a"}`,
    diagnosticsState.rag.lastError ? `error=${diagnosticsState.rag.lastError}` : "",
  ]
    .filter(Boolean)
    .join(" ");

  const perfLine = [
    "[performance]",
    `total_ms=${diagnosticsState.performance.totalLatencyMs ?? "n/a"}`,
    `rag_ms=${diagnosticsState.performance.ragLatencyMs ?? "n/a"}`,
    `llm_ms=${diagnosticsState.performance.llmLatencyMs ?? "n/a"}`,
    `prompt_tokens=${diagnosticsState.performance.promptTokens ?? "n/a"}`,
    `completion_tokens=${diagnosticsState.performance.completionTokens ?? "n/a"}`,
    `tokens_per_sec=${diagnosticsState.performance.tokensPerSecond ?? "n/a"}`,
  ]
    .filter(Boolean)
    .join(" ");

  fields.diagnostics.textContent = [healthLine, topicsLine, ragLine, perfLine].join("\n");
}

function renderMetrics() {
  if (!fields.metrics) {
    return;
  }
  const perf = diagnosticsState.performance;
  fields.metrics.textContent = [
    `total_latency_ms: ${perf.totalLatencyMs ?? "n/a"}`,
    `rag_latency_ms: ${perf.ragLatencyMs ?? "n/a"}`,
    `llm_latency_ms: ${perf.llmLatencyMs ?? "n/a"}`,
    `prompt_tokens: ${perf.promptTokens ?? "n/a"}`,
    `completion_tokens: ${perf.completionTokens ?? "n/a"}`,
    `total_tokens: ${perf.totalTokens ?? "n/a"}`,
    `tokens_per_second: ${perf.tokensPerSecond ?? "n/a"}`,
  ].join("\n");
}

function buildSparkline(values) {
  if (!values.length) {
    return "n/a";
  }
  const bars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  if (range === 0) {
    return bars[3].repeat(values.length);
  }
  return values
    .map((value) => {
      const normalized = (value - min) / range;
      const index = Math.max(0, Math.min(bars.length - 1, Math.round(normalized * (bars.length - 1))));
      return bars[index];
    })
    .join("");
}

function renderPerformanceChart() {
  if (!fields.performanceChart) {
    return;
  }
  const history = diagnosticsState.performanceHistory;
  if (!history.length) {
    fields.performanceChart.textContent = "No query metrics yet.";
    return;
  }
  const latencySeries = history
    .map((entry) => entry.totalLatencyMs)
    .filter((value) => typeof value === "number");
  const tpsSeries = history
    .map((entry) => entry.tokensPerSecond)
    .filter((value) => typeof value === "number");

  const latest = history[history.length - 1];
  fields.performanceChart.textContent = [
    `queries_tracked: ${history.length}`,
    `latency_ms   ${buildSparkline(latencySeries)}  latest=${latest.totalLatencyMs ?? "n/a"}`,
    `tokens/sec   ${buildSparkline(tpsSeries)}  latest=${latest.tokensPerSecond ?? "n/a"}`,
  ].join("\n");
}

function currentSettingsPayload() {
  const seedValue = fields.seed.value.trim();
  const enabledRagCollections = Array.from(
    fields.ragCollections.querySelectorAll("input[type='checkbox']:checked")
  ).map((input) => input.value);
  diagnosticsState.rag.enabledCollections = enabledRagCollections;
  diagnosticsState.rag.useRag = fields.useRag.checked;
  diagnosticsState.rag.temporaryChat = fields.temporaryChat.checked;
  renderDiagnostics();
  return {
    provider: fields.provider.value,
    llm_base_url: fields.llmBaseUrl.value.trim(),
    model: getCurrentModel(),
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

function getCurrentModel() {
  if (!fields.modelCustom || !fields.modelSelect || !fields.provider) {
    return "";
  }
  if (fields.provider.value === "ollama") {
    if (fields.modelSelect.value === "__custom__") {
      return fields.modelCustom.value.trim();
    }
    return fields.modelSelect.value.trim();
  }
  return fields.modelCustom.value.trim();
}

function populateModelSelect(selectedModel) {
  if (!fields.modelSelect || !fields.modelCustom) {
    return;
  }
  fields.modelSelect.innerHTML = "";
  const models = [...availableModels];
  if (selectedModel && !models.includes(selectedModel)) {
    models.unshift(selectedModel);
  }

  models.forEach((modelName) => {
    const option = document.createElement("option");
    option.value = modelName;
    option.textContent = modelName;
    fields.modelSelect.appendChild(option);
  });

  const customOption = document.createElement("option");
  customOption.value = "__custom__";
  customOption.textContent = "Custom...";
  fields.modelSelect.appendChild(customOption);

  if (selectedModel && models.includes(selectedModel)) {
    fields.modelSelect.value = selectedModel;
    fields.modelCustom.hidden = true;
  } else {
    fields.modelSelect.value = "__custom__";
    fields.modelCustom.value = selectedModel || "";
    fields.modelCustom.hidden = false;
  }
}

function updateModelControls() {
  if (!fields.provider || !fields.modelSelect || !fields.refreshModels || !fields.modelCustom) {
    return;
  }
  const isOllama = fields.provider.value === "ollama";
  fields.modelSelect.hidden = !isOllama;
  fields.refreshModels.hidden = !isOllama;
  if (!isOllama) {
    fields.modelCustom.hidden = false;
    return;
  }
  fields.modelCustom.hidden = fields.modelSelect.value !== "__custom__";
}

async function loadAvailableModels() {
  if (!fields.provider || !fields.llmBaseUrl) {
    return;
  }
  if (fields.provider.value !== "ollama") {
    return;
  }
  const baseUrl = encodeURIComponent(fields.llmBaseUrl.value.trim());
  const response = await fetch(`/api/models?provider=ollama&base_url=${baseUrl}`);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  availableModels = data.models || [];
  const selectedModel = getCurrentModel();
  populateModelSelect(selectedModel);
  updateModelControls();
  if (data.error) {
    showStatus(`Model list fetch issue: ${data.error}`);
  } else {
    showStatus(`Loaded ${availableModels.length} Ollama models.`);
  }
}

function applySettings(settings) {
  if (
    !fields.provider ||
    !fields.llmBaseUrl ||
    !fields.systemPrompt ||
    !fields.temperature ||
    !fields.topP ||
    !fields.maxTokens ||
    !fields.numCtx ||
    !fields.repeatPenalty ||
    !fields.seed
  ) {
    return;
  }
  fields.provider.value = settings.provider;
  fields.llmBaseUrl.value = settings.llm_base_url;
  fields.modelCustom.value = settings.model;
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
  updateModelControls();
  populateModelSelect(settings.model);
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
  if (fields.provider.value === "ollama") {
    try {
      await loadAvailableModels();
    } catch (err) {
      showStatus(`Model list unavailable: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
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
  fields.history.innerHTML = "";
  if (!rows.length) {
    fields.history.textContent = "No chat history yet.";
    return;
  }
  rows.forEach((item) => {
    const wrapper = document.createElement("article");
    wrapper.className = "history-item";

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = `${item.created_at} | ${item.provider}/${item.model}`;
    wrapper.appendChild(meta);

    const qa = document.createElement("pre");
    qa.className = "history-qa";
    qa.textContent = `Q: ${item.question}\n\nA: ${item.answer}`;
    wrapper.appendChild(qa);

    const lastMetrics = item.config_snapshot?.last_metrics;
    if (lastMetrics) {
      const perf = document.createElement("div");
      perf.className = "history-meta";
      perf.textContent =
        `latency=${lastMetrics.total_latency_ms ?? "n/a"}ms, ` +
        `tokens/sec=${lastMetrics.tokens_per_second ?? "n/a"}, ` +
        `chunks=${lastMetrics.context_chunks_used ?? "n/a"}`;
      wrapper.appendChild(perf);
    }

    fields.history.appendChild(wrapper);
  });
}

async function clearHistory() {
  if (!window.confirm("Clear all saved chat history? This cannot be undone.")) {
    return;
  }
  const response = await fetch("/api/history", { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  diagnosticsState.performanceHistory = [];
  renderPerformanceChart();
  await loadHistory();
  showStatus(`History cleared (${data.deleted_rows ?? 0} records).`);
}

async function deleteLatestHistory() {
  const response = await fetch("/api/history/latest", { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  await loadHistory();
  showStatus(`Deleted latest history record (${data.deleted_rows ?? 0} removed).`);
}

function clearPerformanceTrend() {
  diagnosticsState.performanceHistory = [];
  renderPerformanceChart();
  showStatus("Performance trend cleared.");
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
      temporary_chat: fields.temporaryChat.checked,
    }),
  });
  if (!response.ok) {
    const errBody = await response.text();
    fields.response.textContent = `Request failed: ${errBody}`;
    diagnosticsState.rag.lastError = errBody;
    diagnosticsState.rag.lastAskedAt = isoNow();
    diagnosticsState.performance = {
      totalLatencyMs: null,
      ragLatencyMs: null,
      llmLatencyMs: null,
      promptTokens: null,
      completionTokens: null,
      totalTokens: null,
      tokensPerSecond: null,
    };
    renderDiagnostics();
    renderMetrics();
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
  diagnosticsState.rag.temporaryChat = fields.temporaryChat.checked;
  diagnosticsState.performance = {
    totalLatencyMs: data.metrics?.total_latency_ms ?? null,
    ragLatencyMs: data.metrics?.rag_latency_ms ?? null,
    llmLatencyMs: data.metrics?.llm_latency_ms ?? null,
    promptTokens: data.metrics?.prompt_tokens ?? null,
    completionTokens: data.metrics?.completion_tokens ?? null,
    totalTokens: data.metrics?.total_tokens ?? null,
    tokensPerSecond: data.metrics?.tokens_per_second ?? null,
  };
  diagnosticsState.performanceHistory.push({
    totalLatencyMs: diagnosticsState.performance.totalLatencyMs,
    tokensPerSecond: diagnosticsState.performance.tokensPerSecond,
  });
  if (diagnosticsState.performanceHistory.length > PERFORMANCE_HISTORY_LIMIT) {
    diagnosticsState.performanceHistory = diagnosticsState.performanceHistory.slice(-PERFORMANCE_HISTORY_LIMIT);
  }
  renderDiagnostics();
  renderMetrics();
  renderPerformanceChart();
  showStatus(
    `Done. Context chunks used: ${data.context.length}. Latency: ${data.metrics?.total_latency_ms ?? "n/a"} ms${fields.temporaryChat.checked ? " (temporary, not saved)" : ""}`
  );
  await loadHistory();
}

bindClick("save-settings", async () => {
  try {
    await saveSettings();
    showStatus("Settings saved.");
  } catch (err) {
    showStatus(err instanceof Error ? err.message : String(err));
  }
});

bindClick("ask", async () => {
  try {
    await askQuestion();
  } catch (err) {
    showStatus(err instanceof Error ? err.message : String(err));
  }
});

bindClick("refresh-topics", async () => {
  showStatus("Refreshing topics...");
  await loadTopics();
  showStatus("Topics refreshed.");
});

bindClick("copy-diagnostics", async () => {
  try {
    await copyDiagnostics();
  } catch (err) {
    showStatus(`Copy failed: ${err instanceof Error ? err.message : String(err)}`);
  }
});

bindClick("clear-history", async () => {
  try {
    await clearHistory();
  } catch (err) {
    showStatus(`Clear history failed: ${err instanceof Error ? err.message : String(err)}`);
  }
});

bindClick("delete-latest-history", async () => {
  try {
    await deleteLatestHistory();
  } catch (err) {
    showStatus(`Delete latest failed: ${err instanceof Error ? err.message : String(err)}`);
  }
});

bindClick("clear-performance-trend", () => {
  clearPerformanceTrend();
});

if (fields.provider) {
  fields.provider.addEventListener("change", toggleProviderFields);
  fields.provider.addEventListener("change", async () => {
    updateModelControls();
    if (fields.provider.value === "ollama") {
      try {
        await loadAvailableModels();
      } catch (err) {
        showStatus(`Model list unavailable: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  });
}
if (fields.modelSelect) {
  fields.modelSelect.addEventListener("change", () => {
    updateModelControls();
  });
}
if (fields.refreshModels) {
  fields.refreshModels.addEventListener("click", async () => {
    try {
      await loadAvailableModels();
    } catch (err) {
      showStatus(`Refresh models failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });
}
if (fields.useRag) {
  fields.useRag.addEventListener("change", () => {
    diagnosticsState.rag.useRag = fields.useRag.checked;
    renderDiagnostics();
  });
}
if (fields.temporaryChat) {
  fields.temporaryChat.addEventListener("change", () => {
    diagnosticsState.rag.temporaryChat = fields.temporaryChat.checked;
    renderDiagnostics();
  });
}

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

async function loadMeta() {
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) {
      throw new Error(await response.text());
    }
    const meta = await response.json();
    const appName = meta.app_name || "Chat Hacksman";
    const version = meta.version || "unknown";
    const repoUrl = meta.repo_url || "";
    if (!fields.footerMeta) {
      return;
    }
    if (repoUrl) {
      fields.footerMeta.innerHTML =
        `${appName} v${version} - ` +
        `<a href="${repoUrl}" target="_blank" rel="noopener noreferrer">GitHub</a>`;
    } else {
      fields.footerMeta.textContent = `${appName} v${version}`;
    }
  } catch (err) {
    if (fields.footerMeta) {
      fields.footerMeta.textContent = "Chat Hacksman";
    }
    showStatus(`Meta load failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function bindClick(id, handler) {
  const node = el(id);
  if (!node) {
    return;
  }
  node.addEventListener("click", handler);
}

async function bootstrap() {
  initHelpTips();
  renderDiagnostics();
  renderMetrics();
  renderPerformanceChart();
  await loadMeta();
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
