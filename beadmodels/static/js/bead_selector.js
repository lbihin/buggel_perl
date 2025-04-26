document.addEventListener('DOMContentLoaded', function () {
    const useCustomBeadsCheckbox = document.getElementById('useCustomBeads');
    const beadSelectionContainer = document.getElementById('beadSelectionContainer');

    if (useCustomBeadsCheckbox && beadSelectionContainer) {
        useCustomBeadsCheckbox.addEventListener('change', function () {
            if (this.checked) {
                new bootstrap.Collapse(beadSelectionContainer).show();
            } else {
                new bootstrap.Collapse(beadSelectionContainer).hide();
            }
        });
    }
});
