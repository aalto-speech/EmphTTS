const seedContainer = document.querySelector("#seedtts-samples");
const tinystressContainer = document.querySelector("#tinystress-samples");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function emphasisMarkup(text) {
  return escapeHtml(text).replace(/\*([^*]+)\*/g, "<strong>$1</strong>");
}

function modelLabel(label) {
  const paperModels = {
    "M_TTS-SFT": ["\\mathcal{M}_{\\mathrm{TTS-SFT}}", "ℳ₍TTS-SFT₎"],
    "M_TTS-SFT (s=0.9)": ["\\mathcal{M}_{\\mathrm{TTS-SFT}}\\;(s = 0.9)", "ℳ₍TTS-SFT₎ (s = 0.9)"],
    "M_TTS-DP-SFT": ["\\mathcal{M}_{\\mathrm{TTS-DP-SFT}}", "ℳ₍TTS-DP-SFT₎"],
  };
  const model = paperModels[label];
  return model ? `<span class=\"paper-math\" data-tex=\"${model[0]}\">${model[1]}</span>` : escapeHtml(label);
}

function audioPlayer(audio) {
  return `<audio controls preload="none" src="${escapeHtml(audio.src)}"></audio>`;
}

function sampleBlock(sample, kind) {
  const text = kind === "tinystress" ? emphasisMarkup(sample.text) : escapeHtml(sample.text);
  const headers = sample.audio
    .map((audio) => `<th scope="col">${modelLabel(audio.label)}</th>`)
    .join("");
  const players = sample.audio
    .map((audio) => `<td>${audioPlayer(audio)}</td>`)
    .join("");

  return `
    <article class="sample-block">
      <div class="target-row">
        <div class="sample-index">Sample ${String(sample.number).padStart(2, "0")}</div>
        <p>${text}</p>
      </div>
      <div class="table-wrap">
        <table class="demo-table">
          <thead>
            <tr>${headers}</tr>
          </thead>
          <tbody>
            <tr>${players}</tr>
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderSamples(container, samples, kind) {
  container.innerHTML = samples.map((sample) => sampleBlock(sample, kind)).join("");
  if (window.katex) {
    container.querySelectorAll(".paper-math").forEach((element) => {
      window.katex.render(element.dataset.tex, element, { throwOnError: false });
    });
  }
}

async function loadSamples() {
  const response = await fetch("assets/samples.json");
  if (!response.ok) {
    throw new Error(`Could not load assets/samples.json: ${response.status}`);
  }
  const data = await response.json();
  renderSamples(seedContainer, data.seedtts, "seedtts");
  renderSamples(tinystressContainer, data.tinystress, "tinystress");
}

loadSamples().catch((error) => {
  const message = `<p class="load-error">${escapeHtml(error.message)}</p>`;
  seedContainer.innerHTML = message;
  tinystressContainer.innerHTML = message;
});
