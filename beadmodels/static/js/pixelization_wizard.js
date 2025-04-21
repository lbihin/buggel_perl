document.addEventListener('DOMContentLoaded', function () {
    const boardRadios = document.querySelectorAll('input[name="board_id"]');
    const gridWidthInput = document.getElementById('id_grid_width');
    const gridHeightInput = document.getElementById('id_grid_height');
    const displayWidth = document.getElementById('display_width');
    const displayHeight = document.getElementById('display_height');

    function updateGridDimensions() {
        const selectedBoard = document.querySelector('input[name="board_id"]:checked');
        if (selectedBoard) {
            const width = selectedBoard.dataset.width;
            const height = selectedBoard.dataset.height;

            gridWidthInput.value = width;
            gridHeightInput.value = height;
            displayWidth.textContent = width;
            displayHeight.textContent = height;
        }
    }

    updateGridDimensions();

    boardRadios.forEach(radio => {
        radio.addEventListener('change', updateGridDimensions);
    });
});
