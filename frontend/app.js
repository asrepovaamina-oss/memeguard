import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.1/dist/index.min.js";
import { BACKEND_URL, REPO_URL } from "./config.js";
import { RESULTS } from "./results.js";

const $ = (id) => document.getElementById(id);

const ATTACK_RU = {
  none: "нет", dehumanizing: "дегуманизация", inferiority: "утверждение неполноценности",
  exclusion: "призыв к исключению", mocking_hate_crime: "насмешка над преступлением ненависти",
  slur: "оскорбительное обозначение группы", stereotype: "стереотип", other: "другое",
};
const CONF_RU = { low: "низкая", medium: "средняя", high: "высокая" };

let clientPromise = null;
let currentFile = null;

function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(BACKEND_URL);
    // если сервер недоступен, при следующей попытке подключаюсь заново
    clientPromise.catch(() => { clientPromise = null; });
  }
  return clientPromise;
}

function setStatus(text, loading = false) {
  $("status").textContent = text;
  $("status").classList.toggle("loading", loading);
  $("status").hidden = !text;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s ?? "";
  return d.innerHTML;
}

function meter(id, value, threshold, on) {
  $(`${id}-fill`).style.width = `${Math.round(value * 100)}%`;
  $(`${id}-fill`).classList.toggle("bad", on);
  $(`${id}-thr`).style.left = `${threshold * 100}%`;
  $(`${id}-val`).textContent = `${Math.round(value * 100)}%`;
}

// штамп поверх мема: пересоздаю класс, чтобы анимация проигрывалась заново
function slam(offensive) {
  const stamp = $("stamp");
  stamp.hidden = true;
  stamp.textContent = offensive ? "оскорбительный" : "норм";
  stamp.className = `stamp ${offensive ? "bad" : "ok"}`;
  void stamp.offsetWidth;
  stamp.hidden = false;
}

function render(r, final) {
  $("score").hidden = false;
  $("verdict").textContent = r.offensive ? "Оскорбительный" : "Нормальный";
  $("verdict").className = `badge ${r.offensive ? "bad" : "ok"}`;
  $("reasons").textContent = r.reasons.length ? r.reasons.join(" + ") : "";

  // у ненависти порог 0.5, когда есть оценка VLM (ансамбль), иначе порог классификатора
  const hateThr = r.vlm_score != null ? 0.5 : r.threshold;
  meter("hate", r.hate_score, hateThr, r.hate);
  meter("rude", r.rude_score, 0.5, r.rude);
  $("hate-src").textContent = r.vlm_score != null ? "классификатор + VLM" : "классификатор";
  $("rude-src").textContent = r.vlm_rude ? "модель токсичности + VLM" : "модель токсичности";

  $("text").value = r.text;
  $("text-source").textContent = r.text_source === "VLM"
    ? "Текст уточнён VLM: OCR прочитал его с ошибками" : `Источник текста: ${r.text_source}`;
  const translated = r.text_en && r.text_en !== r.text;
  $("translated").hidden = !translated;
  if (translated) $("translated").textContent = `Перевод для CLIP: ${r.text_en}`;
  if (final) slam(r.offensive);
}

function renderExplanation(r) {
  const e = r.explanation;
  const box = $("explain");
  if (!e) { box.hidden = true; return; }
  box.hidden = false;
  if (e.error) {
    box.innerHTML = `<p class="hint">Объяснение недоступно: ${esc(e.error)}</p>`;
    return;
  }
  const vlmHate = e.verdict === "hateful";
  const groups = e.target_groups.length
    ? e.target_groups.map((g) => `<span class="chip">${esc(g)}</span>`).join("")
    : "нет";
  box.innerHTML = `
    <h3>Мнение VLM: ${vlmHate ? "ненависть к группе" : "ненависти нет"}${e.rude ? ", грубый" : ""}
      <span class="hint">(уверенность ${CONF_RU[e.confidence] ?? e.confidence})</span></h3>
    ${vlmHate !== (r.prob >= r.threshold)
      ? '<div class="warn">VLM и классификатор расходятся в оценке ненависти. Такой мем стоит проверить вручную.</div>' : ""}
    <dl>
      <dt>Что на картинке</dt><dd>${esc(e.image_description)}</dd>
      <dt>Что значит текст</dt><dd>${esc(e.text_meaning)}</dd>
      <dt>Атакуемые группы</dt><dd>${groups}</dd>
      <dt>Тип атаки</dt><dd>${esc(ATTACK_RU[e.attack_type] ?? e.attack_type)}</dd>
      <dt>Почему</dt><dd>${esc(e.explanation)}</dd>
    </dl>`;
}

async function run(text = "") {
  if (!currentFile) return;
  $("rescore").disabled = true;
  if (!text) {
    $("score").hidden = true;
    $("heat-wrap").hidden = true;
    $("explain").hidden = true;
    $("stamp").hidden = true;
  }
  setStatus("Подключаюсь к модели...", true);

  try {
    const client = await getClient();
    setStatus(text ? "Пересчитываю..." : "Читаю текст и считаю оценки...", true);
    const job = client.submit("/web_analyze", [handle_file(currentFile), text, !text]);

    for await (const msg of job) {
      if (msg.type === "status" && msg.stage === "error") {
        throw new Error(msg.message || "ошибка на сервере");
      }
      if (msg.type !== "data") continue;
      const [r, heat] = msg.data;
      render(r, r.stage === "done");
      if (heat?.url) {
        $("heat").src = heat.url;
        $("heat-wrap").hidden = false;
      }
      if (r.stage === "score") setStatus("Строю карту значимости...", true);
      if (r.stage === "saliency") setStatus(text ? "" : "VLM читает и анализирует мем...", !text);
      if (r.stage === "done") {
        setStatus("");
        if (!text) renderExplanation(r);
        loadStats();
      }
    }
  } catch (err) {
    console.error(err);
    setStatus(`Не получилось связаться с моделью (${err.message}). `
      + "Сервер с моделью запускается на время демонстрации, возможно, сейчас он выключен.");
  } finally {
    $("rescore").disabled = false;
  }
}

function pick(file) {
  if (!file || !file.type.startsWith("image/")) return;
  currentFile = file;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("drop-hint").hidden = true;
  run();
}

$("file").addEventListener("change", (e) => pick(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
$("rescore").addEventListener("click", () => run($("text").value.trim() || " "));

// столбики результатов
const max = Math.max(...RESULTS.map((r) => r.auroc));
$("bars").innerHTML = RESULTS.map((r) => `
  <div class="bar-row ${r.auroc === max ? "best" : ""}">
    <span>${esc(r.name)}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${((r.auroc - 0.5) / 0.35) * 100}%"></div></div>
    <span class="bar-val">${r.auroc.toFixed(3)}</span>
  </div>`).join("") + '<p class="hint">Шкала столбиков начинается с 0.5 (случайное угадывание) и идёт до 0.85.</p>';

$("space-link").href = BACKEND_URL;
$("repo-link").href = REPO_URL;

// мониторинг: сводка из журнала запросов на сервере
async function loadStats() {
  try {
    const client = await getClient();
    const { data: [s] } = await client.predict("/stats", []);
    if (!s.requests) {
      $("stats").innerHTML = '<p class="hint">Запросов пока не было. Загрузите мем выше.</p>';
      return;
    }
    const sec = (ms) => (ms == null ? "нет" : `${(ms / 1000).toFixed(1)} с`);
    const pct = (x) => (x == null ? "нет" : `${Math.round(x * 100)}%`);
    const cards = [
      [s.requests, "запросов к модели"],
      [sec(s.total_median_ms), `медиана времени ответа (95% быстрее ${sec(s.total_p95_ms)})`],
      [pct(s.vlm_error_rate), `ошибок VLM из ${s.vlm_calls} вызовов`],
      [pct(s.disagreement_rate), "мемов, где VLM и классификатор расходятся"],
      [pct(s.ocr_fixed_rate), "мемов, где VLM исправил текст OCR"],
      [pct(s.offensive_rate), "признаны оскорбительными"],
      [sec(s.steps.vlm?.median_ms), "медиана VLM, самый долгий шаг"],
      [sec(s.steps.saliency?.median_ms), "медиана Grad-CAM"],
    ];
    $("stats").innerHTML = cards.map(([v, t]) => `<div class="stat"><b>${esc(String(v))}</b><span>${esc(t)}</span></div>`).join("");
  } catch {
    $("stats").innerHTML = '<p class="hint">Статистика доступна, когда сервер с моделью запущен.</p>';
  }
}

// подключаюсь заранее, пока человек выбирает картинку
getClient().catch(() => {});
loadStats();
