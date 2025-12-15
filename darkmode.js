document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("theme-toggle");
    const html = document.documentElement;

    // Load saved theme
    const savedTheme = localStorage.getItem("theme") || "light";
    html.setAttribute("data-theme", savedTheme);

    toggle.addEventListener("click", () => {
        const current = html.getAttribute("data-theme");
        const newTheme = current === "dark" ? "light" : "dark";
        html.setAttribute("data-theme", newTheme);
        localStorage.setItem("theme", newTheme);
    });
});