(function () {
  function qs(sel, root) { return (root || document).querySelector(sel); }
  function qsa(sel, root) { return Array.from((root || document).querySelectorAll(sel)); }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  // ── Theme ──────────────────────────────────────────────────────────────────
  const docRoot = document.documentElement;
  const savedTheme = localStorage.getItem("predannpp_theme");
  docRoot.setAttribute("data-theme", (savedTheme === "light" || savedTheme === "dark") ? savedTheme : "dark");

  const themeBtn = qs("#themeToggle");
  if (themeBtn) {
    function applyTheme(t) {
      docRoot.setAttribute("data-theme", t);
      localStorage.setItem("predannpp_theme", t);
      themeBtn.setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
      themeBtn.textContent = t === "dark" ? "🌙 Dark" : "☀️ Light";
    }
    applyTheme(docRoot.getAttribute("data-theme") || "dark");
    themeBtn.addEventListener("click", function () {
      applyTheme(docRoot.getAttribute("data-theme") === "dark" ? "light" : "dark");
    });
  }

  // ── Missing-media placeholders ─────────────────────────────────────────────
  function replaceWithPlaceholder(el, msg) {
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = msg;
    el.replaceWith(ph);
  }
  qsa("img[data-fallback]").forEach(img => {
    img.addEventListener("error", () => replaceWithPlaceholder(img, img.getAttribute("data-fallback") || "Missing image."), { once: true });
  });
  qsa("video[data-fallback]").forEach(v => {
    v.addEventListener("error", () => replaceWithPlaceholder(v, v.getAttribute("data-fallback") || "Missing video."), { once: true });
  });
  qsa("audio[data-fallback]").forEach(a => {
    a.addEventListener("error", () => replaceWithPlaceholder(a, a.getAttribute("data-fallback") || "Missing audio."), { once: true });
  });

  // ── Copy buttons ───────────────────────────────────────────────────────────
  qsa("[data-copy-target]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const pre = qs("#" + btn.getAttribute("data-copy-target"));
      if (!pre) { return; }
      try {
        await navigator.clipboard.writeText(pre.textContent || "");
        const old = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => { btn.textContent = old; }, 1200);
      } catch (_e) {
        btn.textContent = "Copy failed";
        setTimeout(() => { btn.textContent = "Copy"; }, 1200);
      }
    });
  });

  // ── Footer year ────────────────────────────────────────────────────────────
  const yearEl = qs("#year");
  if (yearEl) { yearEl.textContent = String(new Date().getFullYear()); }

  // ── "Coming soon" buttons ──────────────────────────────────────────────────
  qsa("[data-soon='true']").forEach(el => {
    el.addEventListener("click", ev => ev.preventDefault());
  });

  // ── Binary search helper ───────────────────────────────────────────────────
  function nearestIndex(arr, target) {
    if (!arr || arr.length === 0) { return 0; }
    let lo = 0, hi = arr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < target) { lo = mid + 1; }
      else if (arr[mid] > target) { hi = mid - 1; }
      else { return mid; }
    }
    if (lo <= 0) { return 0; }
    if (lo >= arr.length) { return arr.length - 1; }
    return Math.abs(arr[lo - 1] - target) <= Math.abs(arr[lo] - target) ? lo - 1 : lo;
  }

  // ── Synchronized feature visualization (generic) ───────────────────────────
  // config = {
  //   canvasSel, audioSel, playBtnSel, sourceSel, featSelSel,
  //   seekSel, seekLblSel, zoomSel, zoomLblSel, statusSel, attrSel,
  //   manifestUrl,      // path to JSON manifest with track list
  //   initialTrackId,   // optional
  //   mfHz,             // optional, default 50
  //   hardCapSec,       // optional, default 240
  // }
  function initSyncViz(config) {
    const canvas = qs(config.canvasSel); if (!canvas) { return; }
    const audio = qs(config.audioSel); if (!audio) { return; }
    const playBtn = qs(config.playBtnSel); if (!playBtn) { return; }
    const sourceSel = qs(config.sourceSel); if (!sourceSel) { return; }
    const featSel = qs(config.featSelSel); if (!featSel) { return; }
    const seekEl = qs(config.seekSel); if (!seekEl) { return; }
    const seekLbl = qs(config.seekLblSel); if (!seekLbl) { return; }
    const zoomEl = qs(config.zoomSel); if (!zoomEl) { return; }
    const zoomLbl = qs(config.zoomLblSel); if (!zoomLbl) { return; }
    const statusEl = qs(config.statusSel); if (!statusEl) { return; }
    const attrEl = qs(config.attrSel);

    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) { statusEl.textContent = "Canvas unavailable."; return; }

    // ── State ────────────────────────────────────────────────────────────────
    const MF_HZ = config.mfHz || 50;       // feature frames per second
    const BASE_WIN = 30;                   // window width (seconds) at 100% effective zoom
    const HARD_CAP_SEC = config.hardCapSec || 240;
    let data = null;                       // loaded track JSON
    let dataEndSec = 0;
    let tracks = [];
    let activeTrack = null;
    let durationCapSec = HARD_CAP_SEC;
    let feature = "surp";// "surp" | "ent"
    let zoomPct = 25;    // 1–100 UI value
    let viewStart = 0;     // left edge of visible window (seconds)
    let rafId = null;
    let isSeeking = false; // user is dragging slider or canvas
    let seekTarget = 0;     // preview position while dragging
    let canvasDrag = false;
    let wasPlayingBeforeSeek = false; // was audio playing when drag started?

    // ── Helpers ──────────────────────────────────────────────────────────────
    function getDuration() {
      const d = Number(audio.duration);
      const audioDur = (Number.isFinite(d) && d > 0) ? d : Infinity;
      const dataDur = (dataEndSec > 0) ? dataEndSec : Infinity;
      const capDur = (durationCapSec > 0) ? durationCapSec : Infinity;
      const dur = Math.min(audioDur, dataDur, capDur);
      return (Number.isFinite(dur) && dur > 0) ? dur : 0;
    }
    function fmt(s) {
      s = Math.max(0, Math.floor(Number(s) || 0));
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }
    function getWinLen() {
      const dur = getDuration();
      if (dur <= 0) { return BASE_WIN; }
      const eff = 1 + (zoomPct - 1) / 99 * 399; // UI 1–100 → effective 1–400×
      return clamp(BASE_WIN * 100 / eff, 0.5, dur);
    }
    function getMaxViewStart() { return Math.max(0, getDuration() - getWinLen()); }
    function centerView(t) { viewStart = clamp(t - getWinLen() * 0.5, 0, getMaxViewStart()); }

    function renderAttribution(track) {
      if (!attrEl || !track) { return; }
      const sourceUrl = track.source_url || "";
      const licenseUrl = track.license_url || "";
      const sourceLink = sourceUrl ? `<a href="${sourceUrl}" target="_blank" rel="noopener">${sourceUrl}</a>` : "N/A";
      const licenseLink = licenseUrl ? `<a href="${licenseUrl}" target="_blank" rel="noopener">${licenseUrl}</a>` : "N/A";
      attrEl.innerHTML = `<strong>${track.display_title || track.id || "Track"}</strong><br>` +
        `${track.artist_credit || ""}<br>` +
        `Source: ${sourceLink}<br>` +
        `License: ${track.license_name || ""} (${licenseLink})`;
    }

    // ── Seek ─────────────────────────────────────────────────────────────────
    // Simply sets audio.currentTime and re-centers the view.
    // Works for both seek-while-paused and seek-while-playing.
    function doSeek(t) {
      const dur = getDuration();
      if (dur <= 0) { return; }
      t = clamp(t, 0, dur);
      audio.currentTime = t;
      centerView(t);
      updateUI();
      draw();
    }

    // Called when the user finishes dragging (slider pointerup / canvas pointerup)
    function commitSeek() {
      isSeeking = false;
      doSeek(seekTarget);
      // Only resume if audio was playing before the drag started
      if (wasPlayingBeforeSeek) {
        audio.play().then(() => startAnim()).catch(() => { stopAnim(); });
      }
      wasPlayingBeforeSeek = false;
    }

    // ── UI update ────────────────────────────────────────────────────────────
    function updateUI() {
      const dur = getDuration();
      const cur = isSeeking ? seekTarget : (Number(audio.currentTime) || 0);

      seekEl.max = dur > 0 ? dur.toFixed(2) : "0";
      if (!isSeeking) { seekEl.value = cur.toFixed(2); }
      seekLbl.textContent = `${fmt(cur)} / ${fmt(dur)}`;

      zoomEl.value = String(Math.round(zoomPct));
      zoomLbl.textContent = `${Math.round(zoomPct)}%`;

      playBtn.textContent = audio.paused ? "▶" : "❚❚";
      playBtn.setAttribute("aria-label", audio.paused ? "Play" : "Pause");
    }

    // ── Draw ─────────────────────────────────────────────────────────────────
    function buildSeries() {
      if (!data || !Array.isArray(data.start_s)) { return []; }
      const key = feature + "_q";   // JSON key convention: <featureValue>_q
      const startArr = data.start_s;
      const matrix = Array.isArray(data[key]) ? data[key] : [];
      if (!matrix.length || !startArr.length) { return []; }
      const winLen = getWinLen();
      const N = Math.max(2, Math.round(winLen * MF_HZ));
      const out = new Array(N);
      for (let i = 0; i < N; i++) {
        const t = viewStart + i / MF_HZ;
        const seg = nearestIndex(startArr, t);
        const row = matrix[seg] || [];
        if (!row.length) {
          out[i] = 0;
          continue;
        }
        const off = clamp(Math.round((t - startArr[seg]) * MF_HZ), 0, row.length - 1);
        out[i] = Number(row[off] ?? 0);
      }
      return out;
    }

    function draw() {
      const W = canvas.clientWidth, H = canvas.clientHeight;
      if (W <= 0 || H <= 0) { return; }
      ctx2d.clearRect(0, 0, W, H);

      const PL = 46, PR = 10, PT = 14, PB = 28;
      const pw = Math.max(10, W - PL - PR);
      const ph = Math.max(10, H - PT - PB);

      // Border
      ctx2d.strokeStyle = "rgba(255,255,255,0.18)";
      ctx2d.lineWidth = 1;
      ctx2d.strokeRect(PL, PT, pw, ph);

      // Grid (¼, ½, ¾)
      ctx2d.strokeStyle = "rgba(255,255,255,0.12)";
      ctx2d.lineWidth = 1;
      for (let g = 1; g < 4; g++) {
        const y = PT + ph * g / 4;
        ctx2d.beginPath(); ctx2d.moveTo(PL, y); ctx2d.lineTo(PL + pw, y); ctx2d.stroke();
      }

      // Feature curve
      const series = buildSeries();
      if (series.length >= 2) {
        const rs = getComputedStyle(document.documentElement);
        ctx2d.strokeStyle = feature === "ent"
          ? (rs.getPropertyValue("--accent2") || "#25d3ff").trim()
          : (rs.getPropertyValue("--accent") || "#7c5cff").trim();
        ctx2d.lineWidth = 1.2;
        ctx2d.beginPath();
        for (let i = 0; i < series.length; i++) {
          const x = PL + (i / (series.length - 1)) * pw;
          const y = PT + (1 - series[i] / 127) * ph;
          i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
        }
        ctx2d.stroke();
      }

      // Y-axis labels
      ctx2d.fillStyle = "rgba(255,255,255,.85)";
      ctx2d.font = "12px sans-serif";
      ctx2d.fillText("127", 12, PT + 4);
      ctx2d.fillText("64", 18, PT + ph * 0.5 + 4);
      ctx2d.fillText("0", 26, PT + ph + 4);

      // X-axis time labels
      const winLen = getWinLen();
      ctx2d.fillText(`${viewStart.toFixed(1)}s`, PL, H - 8);
      ctx2d.fillText(`${(viewStart + winLen).toFixed(1)}s`, PL + pw - 42, H - 8);

      // Red playhead line
      const ph_t = isSeeking ? seekTarget : (Number(audio.currentTime) || 0);
      if (ph_t >= viewStart && ph_t <= viewStart + winLen) {
        const xh = PL + ((ph_t - viewStart) / winLen) * pw;
        ctx2d.strokeStyle = "#ff4d4f";
        ctx2d.lineWidth = 2;
        ctx2d.beginPath(); ctx2d.moveTo(xh, PT); ctx2d.lineTo(xh, PT + ph); ctx2d.stroke();
      }
    }

    // ── Animation loop ────────────────────────────────────────────────────────
    function tick() {
      updateUI(); draw();
      if (!audio.paused && !audio.ended) { rafId = requestAnimationFrame(tick); }
    }
    function startAnim() {
      if (rafId != null) { cancelAnimationFrame(rafId); }
      rafId = requestAnimationFrame(tick);
    }
    function stopAnim() {
      if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
      updateUI(); draw();
    }

    // ── Event listeners ───────────────────────────────────────────────────────

    // Play / Pause button
    playBtn.addEventListener("click", async () => {
      if (!data) { return; }
      if (audio.paused) {
        try { await audio.play(); startAnim(); }
        catch (_) { statusEl.textContent = "Playback failed. Check browser autoplay settings."; }
      } else {
        audio.pause(); stopAnim();
      }
    });

    // Feature select
    featSel.addEventListener("change", () => {
      feature = featSel.value === "ent" ? "ent" : "surp";
      draw();
    });

    // Source select
    sourceSel.addEventListener("change", () => {
      const next = tracks.find(t => t.id === sourceSel.value);
      if (!next) { return; }
      loadTrack(next);
    });

    // ── Seek slider ──────────────────────────────────────────────────────────
    seekEl.addEventListener("pointerdown", () => {
      wasPlayingBeforeSeek = !audio.paused; // record BEFORE pausing
      isSeeking = true;
      seekTarget = clamp(Number(seekEl.value) || 0, 0, getDuration());
      if (wasPlayingBeforeSeek) { audio.pause(); stopAnim(); }
    });
    seekEl.addEventListener("input", () => {
      if (!isSeeking) { return; }
      seekTarget = clamp(Number(seekEl.value) || 0, 0, getDuration());
      centerView(seekTarget);
      updateUI(); draw();
    });
    seekEl.addEventListener("pointerup", commitSeek);
    seekEl.addEventListener("pointercancel", commitSeek);

    // ── Zoom slider ──────────────────────────────────────────────────────────
    zoomEl.addEventListener("input", () => {
      const oldCenter = viewStart + getWinLen() * 0.5;
      zoomPct = Number(zoomEl.value) || 25;
      viewStart = clamp(oldCenter - getWinLen() * 0.5, 0, getMaxViewStart());
      updateUI(); draw();
    });

    // ── Audio events ─────────────────────────────────────────────────────────
    audio.addEventListener("play", () => { startAnim(); updateUI(); });
    audio.addEventListener("pause", () => { stopAnim(); });
    audio.addEventListener("ended", () => { stopAnim(); });
    audio.addEventListener("timeupdate", () => {
      const dur = getDuration();
      if (dur > 0 && audio.currentTime >= dur) {
        audio.currentTime = dur;
        audio.pause();
      }
      if (!isSeeking) { updateUI(); draw(); }
    });
    audio.addEventListener("loadedmetadata", () => { updateUI(); draw(); });

    // ── Canvas: drag / click to seek ─────────────────────────────────────────
    function canvasXToTime(ev) {
      const rect = canvas.getBoundingClientRect();
      const PL = 46, PR = 10;
      const pw = Math.max(10, rect.width - PL - PR);
      const ratio = clamp((ev.clientX - rect.left - PL) / pw, 0, 1);
      return viewStart + ratio * getWinLen();
    }

    // Note: no separate "click" handler — pointerdown/up covers both click and drag.
    canvas.addEventListener("pointerdown", ev => {
      if (!data) { return; }
      wasPlayingBeforeSeek = !audio.paused; // record BEFORE pausing
      canvasDrag = true;
      canvas.setPointerCapture(ev.pointerId);
      isSeeking = true;
      seekTarget = clamp(canvasXToTime(ev), 0, getDuration());
      if (wasPlayingBeforeSeek) { audio.pause(); stopAnim(); }
      updateUI(); draw();
    });
    canvas.addEventListener("pointermove", ev => {
      if (!canvasDrag) { return; }
      seekTarget = clamp(canvasXToTime(ev), 0, getDuration());
      centerView(seekTarget);
      updateUI(); draw();
    });
    canvas.addEventListener("pointerup", ev => {
      if (!canvasDrag) { return; }
      canvasDrag = false;
      canvas.releasePointerCapture(ev.pointerId);
      commitSeek();
    });
    canvas.addEventListener("pointercancel", ev => {
      if (!canvasDrag) { return; }
      canvasDrag = false;
      try { canvas.releasePointerCapture(ev.pointerId); } catch (_) { }
      commitSeek();
    });

    // ── Canvas: wheel to zoom/pan ─────────────────────────────────────────────
    canvas.addEventListener("wheel", ev => {
      if (!data) { return; }
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const PL = 46, PR = 10;
      const pw = Math.max(10, rect.width - PL - PR);
      const ratio = clamp((ev.clientX - rect.left - PL) / pw, 0, 1);
      const anchor = viewStart + ratio * getWinLen();

      if (Math.abs(ev.deltaX) > 0.2 || ev.shiftKey) {
        const pan = (ev.deltaX || ev.deltaY) * 0.01 * Math.max(0.05, getWinLen() * 0.015);
        viewStart = clamp(viewStart + pan, 0, getMaxViewStart());
      } else {
        const effNow = 1 + (zoomPct - 1) / 99 * 399;
        const effNext = clamp(effNow / (ev.deltaY < 0 ? 0.92 : 1.08), 1, 400);
        zoomPct = 1 + (effNext - 1) / 399 * 99;
        viewStart = clamp(anchor - ratio * getWinLen(), 0, getMaxViewStart());
      }
      updateUI(); draw();
    }, { passive: false });

    // Double-click canvas → reset zoom
    canvas.addEventListener("dblclick", () => {
      zoomPct = 25;
      centerView(Number(audio.currentTime) || 0);
      updateUI(); draw();
    });

    // ── Resize ────────────────────────────────────────────────────────────────
    function resizeCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }
    window.addEventListener("resize", resizeCanvas);

    function computeDataEndSec(json) {
      if (!json || !Array.isArray(json.start_s) || json.start_s.length === 0) { return 0; }
      const lastStart = Number(json.start_s[json.start_s.length - 1] || 0);
      const surpLen = Array.isArray(json.surp_q) && json.surp_q.length
        ? (json.surp_q[json.surp_q.length - 1] || []).length
        : 0;
      const entLen = Array.isArray(json.ent_q) && json.ent_q.length
        ? (json.ent_q[json.ent_q.length - 1] || []).length
        : 0;
      const rowLen = Math.max(surpLen, entLen, 0);
      return lastStart + rowLen / MF_HZ;
    }

    async function loadTrack(track) {
      if (!track) { return; }
      activeTrack = track;
      playBtn.disabled = true;
      data = null;
      dataEndSec = 0;
      durationCapSec = Number(track.max_duration_s) || HARD_CAP_SEC;
      feature = featSel.value === "ent" ? "ent" : "surp";
      isSeeking = false;
      seekTarget = 0;
      viewStart = 0;
      statusEl.textContent = `Loading ${track.display_title || track.id}…`;
      renderAttribution(track);

      if (!audio.paused) {
        audio.pause();
      }
      audio.currentTime = 0;
      audio.src = track.audio_url;
      audio.load();

      try {
        const r = await fetch(track.data_url, { cache: "no-store" });
        if (!r.ok) { throw new Error(`HTTP ${r.status}`); }
        const json = await r.json();
        if (!Array.isArray(json.start_s) || !Array.isArray(json.surp_q) || !Array.isArray(json.ent_q)) {
          throw new Error("Invalid JSON schema. Expected start_s, surp_q, ent_q arrays.");
        }
        data = json;
        dataEndSec = computeDataEndSec(json);
        playBtn.disabled = false;
        centerView(0);
        updateUI();
        draw();
        const shownDur = getDuration() || dataEndSec;
        statusEl.textContent = `${track.display_title || track.id} (0:00 – ${fmt(shownDur)}). Press ▶ to play.`;
      } catch (err) {
        statusEl.textContent = `Failed to load track data: ${err.message}`;
      }
    }

    async function loadManifest() {
      statusEl.textContent = "Loading track manifest…";
      playBtn.disabled = true;

      try {
        const r = await fetch(config.manifestUrl, { cache: "no-store" });
        if (!r.ok) { throw new Error(`HTTP ${r.status}`); }
        const manifest = await r.json();
        if (!manifest || !Array.isArray(manifest.tracks) || manifest.tracks.length === 0) {
          throw new Error("Manifest must contain a non-empty tracks array.");
        }

        tracks = manifest.tracks;
        sourceSel.innerHTML = tracks
          .map(t => `<option value="${t.id}">${t.display_title || t.id}</option>`)
          .join("");

        const initialId = config.initialTrackId || tracks[0].id;
        const firstTrack = tracks.find(t => t.id === initialId) || tracks[0];
        sourceSel.value = firstTrack.id;
        await loadTrack(firstTrack);
      } catch (err) {
        statusEl.textContent = `Failed to load manifest: ${err.message}`;
      }
    }

    // Initial canvas sizing
    resizeCanvas();
    loadManifest();
  }

  // ── Initialize visualizations ─────────────────────────────────────────────
  // Add more initSyncViz({...}) calls here for additional players.
  initSyncViz({
    canvasSel: "#surprisalCanvas",
    audioSel: "#syncAudio",
    playBtnSel: "#syncPlayPause",
    sourceSel: "#syncSourceSelect",
    featSelSel: "#syncFeatureSelect",
    seekSel: "#syncSeek",
    seekLblSel: "#syncSeekLabel",
    zoomSel: "#syncZoomPercent",
    zoomLblSel: "#syncZoomPercentLabel",
    statusSel: "#syncStatus",
    attrSel: "#syncAttribution",
    manifestUrl: "./assets/data/manifest.json",
    initialTrackId: "stickybee_josh_woodward",
    hardCapSec: 240,
  });
})();
