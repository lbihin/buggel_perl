document.addEventListener("DOMContentLoaded", function () {
    // Handle form submissions with AJAX
    const forms = document.querySelectorAll("form[data-ajax]");
    forms.forEach((form) => {
        form.addEventListener("submit", function (event) {
            event.preventDefault();
            const formData = new FormData(form);
            fetch(form.action, {
                method: form.method,
                body: formData,
                headers: {
                    "X-CSRFToken": CSRF_TOKEN,
                },
            })
                .then((response) => response.json())
                .then((data) => {
                    if (data.success) {
                        alert("Formulaire soumis avec succès !");
                    } else {
                        alert("Erreur lors de la soumission du formulaire.");
                    }
                })
                .catch((error) => console.error("Erreur:", error));
        });
    });
});
