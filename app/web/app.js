"use strict";
const $ = (id) => document.getElementById(id);
let current = null,
  started = 0,
  timer = null,
  busy = false;
const operations = {
  select: "Listeleme",
  count: "Adet sayımı",
  sum: "Toplam",
  average: "Ortalama",
  min: "En küçük değer",
  max: "En büyük değer",
};
const statuses = {
  valid: "Geçerli sorgu",
  ambiguous: "Belirsiz istek",
  conflicting: "Çelişkili istek",
  insufficient_information: "Eksik bilgi",
};
const operators = {
  eq: "eşittir",
  neq: "eşit değildir",
  gt: "büyüktür",
  gte: "en az",
  lt: "küçüktür",
  lte: "en çok",
  between: "arasında",
  in: "şunlardan biri",
  contains: "içerir",
};
const number = (value) =>
  Number(value).toLocaleString("tr-TR", { maximumFractionDigits: 1 });
function error(message) {
  $("error-banner").textContent = message;
  $("error-banner").hidden = !message;
}
async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : "Girdi veya beklenen JSON geçerli değil. Alanları kontrol edin.",
    );
  }
  return payload;
}
function report(stats) {
  $("report-summary").textContent = stats.reviewed
    ? `${stats.total} deneme · ${stats.reviewed} değerlendirme · İnsan değerlendirmesinde %${number(stats.human_accuracy * 100)} doğru`
    : `${stats.total} deneme · Henüz değerlendirme yok.`;
}
async function health() {
  try {
    const response = await fetch("/health"),
      h = await response.json();
    $("health-dot").className = `dot ${h.ready ? "ready" : "failed"}`;
    $("health-label").textContent = h.ready
      ? "Model hazır"
      : "Model kullanılamıyor";
    $("model-info").textContent =
      `${h.runtime.model || "Model ayarlanmamış"} · ${h.runtime.device || "—"} · ${h.runtime.dtype || "—"}`;
    report(h.evaluation);
    if (!h.ready) error(h.error);
    $("run-button").disabled = !h.ready || busy;
  } catch {
    error("Sunucuya ulaşılamıyor. Bağlantınızı kontrol edin.");
    $("run-button").disabled = true;
  }
}
function addRow(list, name, value) {
  const dt = document.createElement("dt"),
    dd = document.createElement("dd");
  dt.textContent = name;
  dd.textContent = value;
  list.append(dt, dd);
}
function render(result) {
  $("result").hidden = false;
  $("empty-result").hidden = true;
  $("latency").textContent = `${number(result.timing.total_ms / 1000)} sn`;
  $("tokens").textContent = result.timing.generated_tokens;
  $("schema-status").textContent = result.success ? "Uygun" : "Hatalı";
  $("result-state").textContent = result.success
    ? statuses[result.data.status]
    : "Çıktı hatası";
  const node = $("interpretation");
  node.replaceChildren();
  if (result.success) {
    const data = result.data;
    if (data.status !== "valid") {
      const p = document.createElement("p");
      p.className = "clarification";
      p.textContent = data.clarification;
      node.append(p);
    } else {
      const dl = document.createElement("dl");
      addRow(dl, "İşlem", operations[data.operation]);
      addRow(dl, "Hedef", data.target);
      if (data.group_by.length)
        addRow(dl, "Gruplama", data.group_by.join(", "));
      if (data.order_by)
        addRow(
          dl,
          "Sıralama",
          `${data.order_by.field} · ${data.order_by.direction === "desc" ? "Azalan" : "Artan"}`,
        );
      if (data.limit) addRow(dl, "Sonuç sınırı", data.limit);
      node.append(dl);
      data.filters.forEach((filter) => {
        const row = document.createElement("div");
        row.className = "filter-row";
        row.textContent = `${filter.field} ${operators[filter.operator]} ${Array.isArray(filter.value) ? filter.value.join(", ") : filter.value}`;
        node.append(row);
      });
    }
  } else {
    const p = document.createElement("p");
    p.className = "clarification";
    p.textContent = result.error.message;
    node.append(p);
  }
  $("json-output").textContent = JSON.stringify(
    result.data || result.error,
    null,
    2,
  );
  $("comparison").hidden = !result.comparison;
  if (result.comparison) {
    const c = result.comparison;
    $("comparison").textContent =
      `Birebir eşleşme: ${c.exact_match ? "Evet" : "Hayır"} · Yapısal eşleşme: ${c.semantic_match ? "Evet" : "Hayır"}${c.different_fields.length ? " · Farklı alanlar: " + c.different_fields.join(", ") : ""}`;
  }
  const detail = $("timing-detail");
  detail.replaceChildren();
  [
    ["Hazırlama", result.timing.tokenize_ms],
    ["Üretim", result.timing.latency_ms],
    ["Çözümleme", result.timing.decode_ms],
    ["Doğrulama", result.timing.parse_ms],
  ].forEach(([name, value]) => addRow(detail, name, `${number(value)} ms`));
  addRow(detail, "Tekrar denendi", result.retried ? "Evet" : "Hayır");
  $("feedback-message").textContent = "";
  $("feedback-form").hidden = true;
  $("correct-button").classList.remove("selected");
  $("incorrect-button").classList.remove("selected");
  $("note").value = "";
}
$("prompt").addEventListener("input", () => {
  $("char-count").textContent = `${$("prompt").value.length} / 4000`;
});
document.querySelectorAll(".example").forEach((button) =>
  button.addEventListener("click", () => {
    $("prompt").value = button.dataset.prompt;
    $("prompt").dispatchEvent(new Event("input"));
    $("prompt").focus();
  }),
);
$("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    $("query-form").requestSubmit();
  }
});
$("query-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  error("");
  let expected = null;
  if ($("expected").value.trim()) {
    try {
      expected = JSON.parse($("expected").value);
    } catch {
      error("Beklenen JSON sözdizimi geçerli değil.");
      return;
    }
  }
  busy = true;
  current = null;
  $("run-button").disabled = true;
  $("result").hidden = true;
  $("empty-result").hidden = true;
  $("loading-result").hidden = false;
  started = performance.now();
  timer = setInterval(() => {
    $("elapsed").textContent =
      `${number((performance.now() - started) / 1000)} saniye`;
  }, 100);
  try {
    current = await request("/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("prompt").value, expected }),
    });
    render(current);
  } catch (exc) {
    error(exc.message);
    $("empty-result").hidden = false;
  } finally {
    clearInterval(timer);
    busy = false;
    $("loading-result").hidden = true;
    await health();
  }
});
async function feedback(correct) {
  if (!current) return;
  error("");
  const savedId = current.request_id;
  try {
    const response = await request("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request_id: savedId,
        correct,
        category: correct ? null : $("category").value,
        note: correct ? "" : $("note").value,
        expected: current.expected,
      }),
    });
    if (current?.request_id === savedId) {
      $("feedback-message").textContent = "Değerlendirmeniz kaydedildi.";
      $("feedback-form").hidden = true;
      $(correct ? "correct-button" : "incorrect-button").classList.add(
        "selected",
      );
    }
    report(response.evaluation);
  } catch (exc) {
    error(exc.message);
  }
}
$("correct-button").addEventListener("click", () => feedback(true));
$("incorrect-button").addEventListener("click", () => {
  $("feedback-form").hidden = false;
  $("correct-button").classList.remove("selected");
  $("incorrect-button").classList.add("selected");
});
$("feedback-form").addEventListener("submit", (event) => {
  event.preventDefault();
  feedback(false);
});
$("copy-json").addEventListener("click", async (event) => {
  event.preventDefault();
  try {
    await navigator.clipboard.writeText($("json-output").textContent);
    $("copy-json").textContent = "Kopyalandı";
    setTimeout(() => {
      $("copy-json").textContent = "Kopyala";
    }, 1500);
  } catch {
    error(
      "Kopyalama kullanılamıyor; JSON metnini seçerek kopyalayabilirsiniz.",
    );
  }
});
health();
