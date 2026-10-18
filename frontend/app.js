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

function renderScore(r) {
  $("score").hidden = false;
  const pct = Math.round(r.prob * 100);
  $("prob").textContent = `${pct}%`;
  $("verdict").textContent = r.hateful ? "Оскорбительный" : "Не оскорбительный";
  $("verdict").className = `badge ${r.hateful ? "bad" : "ok"}`;
  // штамп поверх мема: пересоздаю класс, чтобы анимация проигрывалась заново
  const stamp = $("stamp");
  stamp.hidden = true;
  stamp.textContent = r.hateful ? "оскорбительный" : "норм";
  stamp.className = `stamp ${r.hateful ? "bad" : "ok"}`;
  void stamp.offsetWidth;
  stamp.hidden = false;
  $("meter-fill").style.width = `${pct}%`;
  $("meter-fill").classList.toggle("bad", r.hateful);
  $("meter-thr").style.left = `${r.threshold * 100}%`;
  $("text").value = r.text;
  const translated = r.text_en && r.text_en !== r.text;
  $("translated").hidden = !translated;
  if (translated) $("translated").textContent = `Перевод для CLIP: ${r.text_en}`;
}

function renderExplanation(e, modelHateful) {
  const box = $("explain");
  if (!e) { box.hidden = true; return; }
  box.hidden = false;
  if (e.error) {
    box.innerHTML = `<p class="hint">Объяснение недоступно: ${esc(e.error)}</p>`;
    return;
  }
  const vlmHateful = e.verdict === "hateful";
  const groups = e.target_groups.length
    ? e.target_groups.map((g) => `<span class="chip">${esc(g)}</span>`).join("")
    : "нет";
  box.innerHTML = `
    <h3>VLM: ${vlmHateful ? "оскорбительный" : "не оскорбительный"}
      <span class="hint">(уверенность ${CONF_RU[e.confidence] ?? e.confidence})</span></h3>
    ${vlmHateful !== modelHateful
      ? '<div class="warn">VLM не согласна с классификатором. Такой мем стоит проверить вручную.</div>' : ""}
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
  }
  setStatus("Подключаюсь к модели...", true);

  try {
    const client = await getClient();
    setStatus(text ? "Пересчитываю..." : "Читаю текст и считаю вероятность...", true);
    const job = client.submit("/web_analyze", [handle_file(currentFile), text, !text]);

    for await (const msg of job) {
      if (msg.type === "status" && msg.stage === "error") {
        throw new Error(msg.message || "ошибка на сервере");
      }
      if (msg.type !== "data") continue;
      const [r, heat] = msg.data;
      renderScore(r);
      if (heat?.url) {
        $("heat").src = heat.url;
        $("heat-wrap").hidden = false;
      }
      if (r.stage === "score") setStatus("Строю карту значимости...", true);
      if (r.stage === "saliency") setStatus(text ? "" : "VLM анализирует мем...", !text);
      if (r.stage === "done") {
        setStatus("");
        if (!text) renderExplanation(r.explanation, r.hateful);
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
  $("stamp").hidden = true;
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
    <div class="bar-track"><div class="bar-fill" style="width:${((r.auroc - 0.5) / 0.3) * 100}%"></div></div>
    <span class="bar-val">${r.auroc.toFixed(3)}</span>
  </div>`).join("") + '<p class="hint">Шкала столбиков начинается с 0.5 (случайное угадывание) и идёт до 0.8.</p>';

$("space-link").href = BACKEND_URL;
$("repo-link").href = REPO_URL;

// подключаюсь заранее, пока человек выбирает картинку
getClient().catch(() => {});
