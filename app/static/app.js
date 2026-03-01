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
  topicButtons: el("topic-buttons"),
  useRag: el("use-rag"),
  numCtxField: el("num-ctx-field"),
  seedField: el("seed-field"),
};

function showStatus(msg) {
  fields.status.textContent = msg;
}

function currentSettingsPayload() {
  const seedValue = fields.seed.value.trim();
  return {
    provider: fields.provider.value,
    llm_base_url: fields.llmBaseUrl.value.trim(),
    model: fields.model.value.trim(),
    system_prompt: fields.systemPrompt.value,
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
    topics.slice(0, 50).forEach((topic) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = topic;
      button.addEventListener("click", async () => {
        fields.question.value = topic;
        await askQuestion();
      });
      fields.topicButtons.appendChild(button);
    });
  } catch (err) {
    showStatus(`Topic load failed: ${err.message}`);
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
    showStatus("Chat request failed.");
    return;
  }
  const data = await response.json();
  fields.response.textContent = data.answer;
  showStatus(`Done. Context chunks used: ${data.context.length}`);
  await loadHistory();
}

el("save-settings").addEventListener("click", async () => {
  try {
    await saveSettings();
    showStatus("Settings saved.");
  } catch (err) {
    showStatus(err.message);
  }
});

el("ask").addEventListener("click", async () => {
  try {
    await askQuestion();
  } catch (err) {
    showStatus(err.message);
  }
});

el("refresh-topics").addEventListener("click", async () => {
  showStatus("Refreshing topics...");
  await loadTopics();
  showStatus("Topics refreshed.");
});

fields.provider.addEventListener("change", toggleProviderFields);

async function bootstrap() {
  await loadSettings();
  await loadTopics();
  await loadHistory();
  showStatus("Ready.");
}

bootstrap().catch((err) => {
  showStatus(`Startup failed: ${err.message}`);
});
