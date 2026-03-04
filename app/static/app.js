const el = (id) => document.getElementById(id);

const fields = {
  provider: el("provider"),
  llmBaseUrl: el("llm-base-url"),
  llmBaseUrlCustom: el("llm-base-url-custom"),
  modelSelect: el("model-select"),
  modelInput: el("model-input"),
  pullModelName: el("pull-model-name"),
  pullModel: el("pull-model"),
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
  responsePanel: el("response-panel"),
  responseLoading: el("response-loading"),
  metrics: el("metrics"),
  performanceChart: el("performance-chart"),
  history: el("history"),
  feedbackList: el("feedback-list"),
  feedbackUp: el("feedback-up"),
  feedbackDown: el("feedback-down"),
  feedbackText: el("feedback-text"),
  submitFeedback: el("submit-feedback"),
  refreshFeedback: el("refresh-feedback"),
  feedbackStatus: el("feedback-status"),
  diagnostics: el("diagnostics"),
  footerMeta: el("footer-meta"),
  topicButtons: el("topic-buttons"),
  ragCollections: el("rag-collections"),
  useRag: el("use-rag"),
  temporaryChat: el("temporary-chat"),
  numCtxField: el("num-ctx-field"),
  seedField: el("seed-field"),
};
const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));
let availableRagCollections = [];
let availableModels = [];
let rememberedLlmBaseUrls = [];
let selectedFeedbackRating = null;
let lastResponseForFeedback = null;
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

function showFeedbackStatus(msg) {
  if (fields.feedbackStatus) {
    fields.feedbackStatus.textContent = msg;
  }
}

function setResponseLoading(isLoading) {
  if (!fields.responseLoading) {
    return;
  }
  fields.responseLoading.hidden = !isLoading;
}

function pulseResponsePanel() {
  if (!fields.responsePanel) {
    return;
  }
  fields.responsePanel.classList.remove("fresh-response");
  // Restart CSS animation for each new response.
  void fields.responsePanel.offsetWidth;
  fields.responsePanel.classList.add("fresh-response");
}

function setActiveTab(tabId) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tabTarget === tabId);
  });
  tabPanels.forEach((panel) => {
    panel.hidden = panel.dataset.tabPanel !== tabId;
  });
}

function getEffectiveLlmBaseUrl() {
  const custom = fields.llmBaseUrlCustom ? fields.llmBaseUrlCustom.value.trim() : "";
  if (custom) {
    return custom;
  }
  return fields.llmBaseUrl ? fields.llmBaseUrl.value.trim() : "";
}

function formatRelativeTime(isoTime) {
  const timestamp = Date.parse(isoTime || "");
  if (Number.isNaN(timestamp)) {
    return "unknown";
  }
  const diffMs = Math.max(0, Date.now() - timestamp);
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) {
    return `${seconds} second${seconds === 1 ? "" : "s"} ago`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  }
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function populateLlmBaseUrlSelect(selectedUrl) {
  if (!fields.llmBaseUrl) {
    return;
  }
  const chosen = (selectedUrl || "").trim();
  const map = new Map();
  rememberedLlmBaseUrls.forEach((item) => {
    if (!item || !item.url) {
      return;
    }
    map.set(item.url, item);
  });
  if (chosen && !map.has(chosen)) {
    map.set(chosen, {
      url: chosen,
      available: null,
      last_changed_at: null,
      last_checked_at: null,
    });
  }
  rememberedLlmBaseUrls = Array.from(map.values()).sort((a, b) =>
    a.url.localeCompare(b.url, undefined, { sensitivity: "base" })
  );

  fields.llmBaseUrl.innerHTML = "";
  rememberedLlmBaseUrls.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.url;
    const statusLabel =
      item.available === true ? "up" : item.available === false ? "down" : "unknown";
    const relative = item.last_changed_at ? formatRelativeTime(item.last_changed_at) : "unknown";
    option.textContent = `${item.url} (${statusLabel}, changed ${relative})`;
    if (item.last_changed_at) {
      option.title = `Last changed: ${new Date(item.last_changed_at).toLocaleString()}`;
    } else {
      option.title = "Last changed: unknown";
    }
    fields.llmBaseUrl.appendChild(option);
  });

  if (chosen && rememberedLlmBaseUrls.some((item) => item.url === chosen)) {
    fields.llmBaseUrl.value = chosen;
  } else if (rememberedLlmBaseUrls.length > 0) {
    fields.llmBaseUrl.value = rememberedLlmBaseUrls[0].url;
  }
}

async function loadLlmBaseUrls(selectedUrl) {
  const response = await fetch("/api/llm-base-urls/status?limit=500");
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  rememberedLlmBaseUrls = Array.isArray(data.items) ? data.items : [];
  populateLlmBaseUrlSelect(selectedUrl || getEffectiveLlmBaseUrl());
}

function initHelpTips() {
  const tipButtons = Array.from(document.querySelectorAll(".help-tip"));
  let tooltip = document.getElementById("floating-help-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "floating-help-tooltip";
    tooltip.className = "floating-tooltip";
    tooltip.hidden = true;
    document.body.appendChild(tooltip);
  }
  let activeButton = null;

  const hideTooltip = () => {
    tooltip.hidden = true;
    if (activeButton) {
      activeButton.setAttribute("aria-expanded", "false");
      activeButton = null;
    }
  };

  const showTooltipForButton = (button) => {
    const text = button.dataset.help || "No help available.";
    tooltip.textContent = text;
    const rect = button.getBoundingClientRect();
    tooltip.hidden = false;

    const margin = 8;
    let top = rect.bottom + margin;
    let left = rect.left;
    const tooltipRect = tooltip.getBoundingClientRect();
    if (left + tooltipRect.width > window.innerWidth - margin) {
      left = Math.max(margin, window.innerWidth - tooltipRect.width - margin);
    }
    if (top + tooltipRect.height > window.innerHeight - margin) {
      top = Math.max(margin, rect.top - tooltipRect.height - margin);
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  };

  const hideAll = () => {
    hideTooltip();
  };

  tipButtons.forEach((button) => {
    if (!button.dataset.help) {
      return;
    }
    button.setAttribute("aria-label", "Show help");
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      if (activeButton === button && !tooltip.hidden) {
        hideTooltip();
        return;
      }
      hideTooltip();
      activeButton = button;
      button.setAttribute("aria-expanded", "true");
      showTooltipForButton(button);
    });
  });

  document.addEventListener("click", hideAll);
  window.addEventListener("resize", hideAll);
  window.addEventListener("scroll", hideAll, true);
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
      llm_base_url: getEffectiveLlmBaseUrl(),
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
    llm_base_url: getEffectiveLlmBaseUrl(),
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
  if (!fields.modelInput || !fields.modelSelect || !fields.provider) {
    return "";
  }
  if (fields.provider.value === "ollama") {
    return fields.modelSelect.value.trim();
  }
  return fields.modelInput.value.trim();
}

function populateModelSelect(selectedModel) {
  if (!fields.modelSelect || !fields.modelInput) {
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

  if (selectedModel && models.includes(selectedModel)) {
    fields.modelSelect.value = selectedModel;
  } else if (models.length > 0) {
    fields.modelSelect.value = models[0];
  } else {
    fields.modelSelect.value = "";
  }
}

function updateModelControls() {
  if (
    !fields.provider ||
    !fields.modelSelect ||
    !fields.refreshModels ||
    !fields.modelInput ||
    !fields.pullModelName ||
    !fields.pullModel
  ) {
    return;
  }
  const isOllama = fields.provider.value === "ollama";
  fields.modelSelect.hidden = !isOllama;
  fields.refreshModels.hidden = !isOllama;
  fields.pullModelName.hidden = !isOllama;
  fields.pullModel.hidden = !isOllama;
  fields.modelInput.hidden = isOllama;
}

async function loadAvailableModels() {
  if (!fields.provider || !fields.llmBaseUrl) {
    return;
  }
  if (fields.provider.value !== "ollama") {
    return;
  }
  const baseUrl = encodeURIComponent(getEffectiveLlmBaseUrl());
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
  populateLlmBaseUrlSelect(settings.llm_base_url);
  if (fields.llmBaseUrlCustom) {
    fields.llmBaseUrlCustom.value = "";
  }
  fields.modelInput.value = settings.model;
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
  try {
    await loadLlmBaseUrls(data.llm_base_url);
  } catch (err) {
    showStatus(`LLM URL list unavailable: ${err instanceof Error ? err.message : String(err)}`);
  }
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

async function pullModel() {
  if (!fields.pullModelName || !fields.llmBaseUrl) {
    return;
  }
  const name = fields.pullModelName.value.trim();
  if (!name) {
    showStatus("Enter a model name to pull.");
    return;
  }

  showStatus(`Pulling model ${name}...`);
  const response = await fetch("/api/models/pull", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: "ollama",
      base_url: getEffectiveLlmBaseUrl(),
      name,
    }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = await response.json();
  await loadLlmBaseUrls(getEffectiveLlmBaseUrl());
  await loadAvailableModels();
  populateModelSelect(data.pulled_model);
  fields.pullModelName.value = "";
  showStatus(`Pulled model ${data.pulled_model}.`);
}

function renderRagCollectionCheckboxes(availableCollections, enabledCollections) {
  fields.ragCollections.innerHTML = "";
  if (!availableCollections.length) {
    fields.ragCollections.textContent =
      "No Qdrant collections configured. Set RAG_COLLECTIONS or RAG_COLLECTION_1..5.";
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

function updateFeedbackButtons() {
  if (fields.feedbackUp) {
    fields.feedbackUp.classList.toggle("selected", selectedFeedbackRating === "up");
  }
  if (fields.feedbackDown) {
    fields.feedbackDown.classList.toggle("selected", selectedFeedbackRating === "down");
  }
}

function setFeedbackRating(rating) {
  selectedFeedbackRating = selectedFeedbackRating === rating ? null : rating;
  updateFeedbackButtons();
}

async function submitFeedback() {
  if (!lastResponseForFeedback) {
    showFeedbackStatus("Ask a question first so there is a response to rate.");
    return;
  }
  if (!selectedFeedbackRating) {
    showFeedbackStatus("Choose thumbs up or thumbs down first.");
    return;
  }

  const payload = {
    rating: selectedFeedbackRating,
    text: fields.feedbackText ? fields.feedbackText.value.trim() : "",
    history_id: lastResponseForFeedback.historyId,
    question: lastResponseForFeedback.question,
    answer: lastResponseForFeedback.answer,
    provider: lastResponseForFeedback.provider,
    model: lastResponseForFeedback.model,
  };
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  if (fields.feedbackText) {
    fields.feedbackText.value = "";
  }
  selectedFeedbackRating = null;
  updateFeedbackButtons();
  showFeedbackStatus("Feedback saved.");
  await loadFeedback();
}

async function loadFeedback() {
  if (!fields.feedbackList) {
    return;
  }
  const response = await fetch("/api/feedback?limit=50");
  if (!response.ok) {
    fields.feedbackList.textContent = "Unable to load feedback.";
    return;
  }
  const rows = await response.json();
  fields.feedbackList.innerHTML = "";
  if (!rows.length) {
    fields.feedbackList.textContent = "No feedback yet.";
    return;
  }

  rows.forEach((item) => {
    const wrapper = document.createElement("article");
    wrapper.className = "history-item";

    const meta = document.createElement("div");
    meta.className = "history-meta";
    const rating = item.rating === "up" ? "👍" : "👎";
    const linkPart = item.history_id ? ` | history_id=${item.history_id}` : "";
    meta.textContent = `${item.created_at} | ${rating} ${item.provider}/${item.model}${linkPart}`;
    wrapper.appendChild(meta);

    const qa = document.createElement("pre");
    qa.className = "history-qa";
    const note = item.text ? `Feedback: ${item.text}\n\n` : "";
    qa.textContent = `${note}Q: ${item.question}\n\nA: ${item.answer}`;
    wrapper.appendChild(qa);

    fields.feedbackList.appendChild(wrapper);
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
  setResponseLoading(true);
  try {
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
      lastResponseForFeedback = null;
      showFeedbackStatus("");
      return;
    }
    const data = await response.json();
    fields.response.textContent = data.answer;
    pulseResponsePanel();
    lastResponseForFeedback = {
      question,
      answer: data.answer,
      historyId: data.history_id ?? null,
      provider: fields.provider ? fields.provider.value : "",
      model: getCurrentModel(),
    };
    selectedFeedbackRating = null;
    updateFeedbackButtons();
    showFeedbackStatus("");
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
    await loadFeedback();
  } finally {
    setResponseLoading(false);
  }
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

bindClick("feedback-up", () => {
  setFeedbackRating("up");
});

bindClick("feedback-down", () => {
  setFeedbackRating("down");
});

bindClick("submit-feedback", async () => {
  try {
    await submitFeedback();
  } catch (err) {
    showFeedbackStatus(`Save feedback failed: ${err instanceof Error ? err.message : String(err)}`);
  }
});

bindClick("refresh-feedback", async () => {
  try {
    await loadFeedback();
    showFeedbackStatus("Feedback list refreshed.");
  } catch (err) {
    showFeedbackStatus(`Refresh feedback failed: ${err instanceof Error ? err.message : String(err)}`);
  }
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
if (fields.llmBaseUrl) {
  fields.llmBaseUrl.addEventListener("change", async () => {
    if (fields.llmBaseUrlCustom) {
      fields.llmBaseUrlCustom.value = "";
    }
    if (fields.provider && fields.provider.value === "ollama") {
      try {
        await loadAvailableModels();
      } catch (err) {
        showStatus(`Model list unavailable: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  });
}
if (fields.llmBaseUrlCustom) {
  fields.llmBaseUrlCustom.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter") {
      return;
    }
    event.preventDefault();
    const customUrl = fields.llmBaseUrlCustom.value.trim();
    if (!customUrl) {
      return;
    }
    populateLlmBaseUrlSelect(customUrl);
    if (fields.provider && fields.provider.value === "ollama") {
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
      showStatus("Refreshing Ollama model list...");
      await loadAvailableModels();
    } catch (err) {
      showStatus(`Refresh models failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });
}
if (fields.pullModel) {
  fields.pullModel.addEventListener("click", async () => {
    try {
      await pullModel();
    } catch (err) {
      showStatus(`Pull model failed: ${err instanceof Error ? err.message : String(err)}`);
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
if (fields.question) {
  fields.question.addEventListener("keydown", async (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
      return;
    }
    event.preventDefault();
    try {
      await askQuestion();
    } catch (err) {
      showStatus(err instanceof Error ? err.message : String(err));
    }
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

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const target = button.dataset.tabTarget;
    if (target) {
      setActiveTab(target);
    }
  });
});

async function bootstrap() {
  setActiveTab("chat");
  initHelpTips();
  renderDiagnostics();
  renderMetrics();
  renderPerformanceChart();
  await loadMeta();
  await checkHealth();
  try {
    await loadLlmBaseUrls();
  } catch (err) {
    showStatus(`LLM URL list unavailable: ${err instanceof Error ? err.message : String(err)}`);
  }
  await loadRagCollections();
  await loadSettings();
  await loadTopics();
  await loadHistory();
  await loadFeedback();
  showStatus("Ready.");
}

bootstrap().catch((err) => {
  showStatus(`Startup failed: ${err.message}`);
});
