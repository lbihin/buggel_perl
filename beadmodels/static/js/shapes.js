document.addEventListener("DOMContentLoaded", function () {
    const shapeTypeSelect = document.getElementById("shapeType");
    const shapeControls = document.getElementById("shapeControls");

    if (shapeTypeSelect && shapeControls) {
        shapeTypeSelect.addEventListener("change", function () {
            const shapeType = shapeTypeSelect.value;
            fetch(`/shapes/${shapeType}/controls/`)
                .then((response) => response.text())
                .then((html) => {
                    shapeControls.innerHTML = html;
                })
                .catch((error) => console.error("Erreur:", error));
        });
    }
});
