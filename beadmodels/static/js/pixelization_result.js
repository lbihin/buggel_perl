document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('downloadBtn').addEventListener('click', function () {
        const imageElement = document.getElementById('pixelizedImage');
        const imageURL = imageElement.getAttribute('src');

        const downloadLink = document.createElement('a');
        downloadLink.href = imageURL;
        downloadLink.download = 'modele_perles.png';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    });
});
