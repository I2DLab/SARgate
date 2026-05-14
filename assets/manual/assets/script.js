/* =================================
   SARgate Manual – Navigation Script
   ================================= */

document.addEventListener("DOMContentLoaded", () => {
    renderSidebarNavigation();
});

const MANUAL_INDEX = [
    ["1_introduction.html", "Introduction"],
    ["2_installation.html", "Installation"],
    ["3_gui-overview.html", "GUI Overview"],
    ["4_input-files-and-settings.html", "Input Files & Settings"],
    ["5_analysis-workflow.html", "Analysis Workflow"],
    ["6_overview.html", "Overview"],
    ["7_similarity.html", "Similarity"],
    ["8_r-analysis.html", "R-Groups Analysis"],
    ["9_stereo.html", "Stereoisomers"],
    ["10_mmpa.html", "MMPA"],
    ["11_chemspace.html", "Chemical Space"],
    ["12_sar-notes.html", "SAR Notes"],
    ["13_prediction.html", "Prediction"],
    ["14_molecule-drawer.html", "Molecule Drawer"],
    ["15_utilities.html", "Utilities"],
    ["16_license.html", "License & Citation"],
];

function renderSidebarNavigation() {
    const sidebar = document.querySelector(".sidebar");
    if (!sidebar) return;

    const current = window.location.pathname.split("/").pop();
    const title = document.createElement("h1");
    title.textContent = "SARgate Manual";

    const list = document.createElement("ul");
    MANUAL_INDEX.forEach(([href, label]) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = href;
        a.textContent = label;
        if (href === current) {
            a.classList.add("active");
        }
        li.appendChild(a);
        list.appendChild(li);
    });

    sidebar.replaceChildren(title, list);
}
