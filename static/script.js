"use strict";

const dosya = document.getElementById("dosya");
const birak = document.getElementById("birak");
const onizleme = document.getElementById("onizleme");
const onizlemeBos = document.getElementById("onizleme-bos");
const taniBtn = document.getElementById("tani-btn");
const form = document.getElementById("form");
const yukleniyor = document.getElementById("yukleniyor");
const sonuc = document.getElementById("sonuc");
const hata = document.getElementById("hata");

function dosyaSecildi(f) {
  if (!f) return;
  onizleme.src = URL.createObjectURL(f);
  onizleme.hidden = false;
  onizlemeBos.hidden = true;
  taniBtn.disabled = false;
}

dosya.addEventListener("change", () => dosyaSecildi(dosya.files[0]));

// Sürükle-bırak
["dragover", "dragenter"].forEach((e) =>
  birak.addEventListener(e, (ev) => {
    ev.preventDefault();
    birak.classList.add("aktif");
  })
);
["dragleave", "drop"].forEach((e) =>
  birak.addEventListener(e, (ev) => {
    ev.preventDefault();
    birak.classList.remove("aktif");
  })
);
birak.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer.files[0];
  if (f) {
    dosya.files = ev.dataTransfer.files;
    dosyaSecildi(f);
  }
});

// Gönder
form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!dosya.files[0]) return;

  hata.hidden = true;
  sonuc.hidden = true;
  yukleniyor.hidden = false;
  taniBtn.disabled = true;

  try {
    const fd = new FormData();
    fd.append("gorsel", dosya.files[0]);
    const yanit = await fetch("/tani", { method: "POST", body: fd });
    const veri = await yanit.json();
    if (!yanit.ok) throw new Error(veri.hata || "Sunucu hatası");
    sonucGoster(veri);
  } catch (e) {
    hata.textContent = "Hata: " + e.message;
    hata.hidden = false;
  } finally {
    yukleniyor.hidden = true;
    taniBtn.disabled = false;
  }
});

function sonucGoster(v) {
  document.getElementById("sonuc-gorsel").src = v.sorgu;
  document.getElementById("secilen-model").textContent = v.vlm.secilen_model;

  const yuzde = Math.round(v.vlm.guven * 100);
  document.getElementById("guven-dolu").style.width = yuzde + "%";
  document.getElementById("guven-yuzde").textContent = "%" + yuzde + " güven";
  document.getElementById("gerekce").textContent = v.vlm.gerekce || "—";

  const grid = document.getElementById("adaylar");
  grid.innerHTML = "";
  v.adaylar.forEach((a) => {
    const skorYuzde = Math.round(a.skor * 100);
    const secili = a.model === v.vlm.secilen_model ? " secilen" : "";
    const kart = document.createElement("div");
    kart.className = "aday-kart" + secili;
    kart.innerHTML =
      '<img src="' + a.gorsel + '" alt="" loading="lazy">' +
      '<div class="aday-bilgi">' +
        "<b>" + a.sira + ". " + a.model + "</b>" +
        '<span class="kategori">' + a.kategori + "</span>" +
        '<div class="skor-bar"><div style="width:' + skorYuzde + '%"></div></div>' +
        "<small>benzerlik " + a.skor.toFixed(4) + "</small>" +
      "</div>";
    grid.appendChild(kart);
  });

  sonuc.hidden = false;
  sonuc.scrollIntoView({ behavior: "smooth", block: "start" });
}
