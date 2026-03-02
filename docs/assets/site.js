(function(){
  function qs(sel, root){ return (root || document).querySelector(sel); }
  function qsa(sel, root){ return Array.from((root || document).querySelectorAll(sel)); }

  // Theme: default = dark, persist in localStorage
  const root = document.documentElement;
  const saved = localStorage.getItem("predannpp_theme");
  if(saved === "light" || saved === "dark"){
    root.setAttribute("data-theme", saved);
  }else{
    root.setAttribute("data-theme", "dark");
  }

  const themeBtn = qs("#themeToggle");
  if(themeBtn){
    themeBtn.addEventListener("click", function(){
      const cur = root.getAttribute("data-theme") || "dark";
      const next = cur === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      localStorage.setItem("predannpp_theme", next);
      themeBtn.setAttribute("aria-label", next === "dark" ? "Switch to light mode" : "Switch to dark mode");
      themeBtn.textContent = next === "dark" ? "🌙 Dark" : "☀️ Light";
    });

    // Set initial label
    const cur = root.getAttribute("data-theme") || "dark";
    themeBtn.textContent = cur === "dark" ? "🌙 Dark" : "☀️ Light";
  }

  // Replace missing images/videos with a nice placeholder
  function replaceWithPlaceholder(el, msg){
    const ph = document.createElement("div");
    ph.className = "placeholder";
    ph.textContent = msg;
    el.replaceWith(ph);
  }

  qsa("img[data-fallback]").forEach(img => {
    img.addEventListener("error", () => {
      replaceWithPlaceholder(img, img.getAttribute("data-fallback") || "Missing image.");
    }, { once:true });
  });

  qsa("video[data-fallback]").forEach(v => {
    v.addEventListener("error", () => {
      replaceWithPlaceholder(v, v.getAttribute("data-fallback") || "Missing video.");
    }, { once:true });
  });

  // Copy buttons for code blocks
  qsa("[data-copy-target]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-copy-target");
      const pre = qs("#" + id);
      if(!pre){ return; }
      const text = pre.textContent || "";
      try{
        await navigator.clipboard.writeText(text);
        const old = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => btn.textContent = old, 1200);
      }catch(e){
        btn.textContent = "Copy failed";
        setTimeout(() => btn.textContent = "Copy", 1200);
      }
    });
  });

  // Update footer year
  const year = qs("#year");
  if(year){
    year.textContent = String(new Date().getFullYear());
  }

  // "Coming soon" buttons tooltip
  qsa("[data-soon='true']").forEach(el => {
    el.addEventListener("click", (ev) => {
      ev.preventDefault();
    });
  });
})();