document.addEventListener("DOMContentLoaded", () => {
  const year = document.querySelector("[data-year]");
  if (year) year.textContent = new Date().getFullYear();

  const toggle = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".primary-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", String(open));
      toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
      nav.classList.toggle("is-open", open);
    });
  }

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", async () => {
      const code = document.getElementById(button.dataset.copy);
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.innerText);
        const previous = button.textContent;
        button.textContent = "✓";
        window.setTimeout(() => { button.textContent = previous; }, 1300);
      } catch {
        button.textContent = "Select code";
        window.setTimeout(() => { button.textContent = "▢"; }, 1300);
      }
    });
  });
});
