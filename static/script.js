// Custom JavaScript for Brain Tumor Detection app

console.log('Brain Tumor Detection app loaded');

document.addEventListener('DOMContentLoaded', function () {
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');

    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');

        if (currentLocation === linkPath) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Show a confirmation dialog before form submission
    const forms = document.querySelectorAll('form.confirmation-required');
    forms.forEach(form => {
        form.addEventListener('submit', function (e) {
            const confirmed = confirm('Are you sure you want to submit this form?');
            if (!confirmed) {
                e.preventDefault();
            }
        });
    });

    // Toggle visibility of elements
    const toggleButtons = document.querySelectorAll('[data-toggle="visibility"]');
    toggleButtons.forEach(button => {
        button.addEventListener('click', function () {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                targetElement.classList.toggle('hidden');
            }
        });
    });

    // File upload and preview functionality
    const dropArea = document.getElementById('drop-area');
    const fileUpload = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const uploadPrompt = document.getElementById('upload-prompt');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadForm = document.getElementById('upload-form');
    const analyzeText = document.getElementById('analyze-text');
    const loadingText = document.getElementById('loading-text');

    if (dropArea && fileUpload) {
        // Click to select file
        dropArea.addEventListener("click", function () {
            fileUpload.click();
        });

        // Handle file selection
        fileUpload.addEventListener("change", function () {
            if (fileUpload.files.length) {
                handleFiles(fileUpload.files);
            }
        });

        // Drag and drop functionality
        ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ["dragenter", "dragover"].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add("bg-light"), false);
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove("bg-light"), false);
        });

        // Handle file drop
        dropArea.addEventListener("drop", function (e) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length) {
                fileUpload.files = files;
                handleFiles(files);
            }
        });

        function handleFiles(files) {
            const file = files[0];

            // Validate file type
            if (!file.type.match("image.*")) {
                alert("Please upload a valid image file (JPG or PNG)");
                return;
            }

            // Show preview
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove("d-none");
                uploadPrompt.classList.add("d-none");
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    }

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener("submit", function (e) {
            if (!fileUpload.files.length) {
                alert("Please select an MRI scan before analyzing.");
                e.preventDefault();
                return;
            }

            analyzeText.classList.add("d-none");
            loadingText.classList.remove("d-none");
            analyzeBtn.disabled = true;

            // Simulate processing
            setTimeout(() => {
                const analyzedImagePath = imagePreview.src;

                const resultsContainer = document.getElementById("results-container");
                const resultImage = document.getElementById("result-image");
                resultImage.src = analyzedImagePath;
                resultsContainer.classList.remove("d-none");

                loadingText.classList.add("d-none");
                analyzeText.classList.remove("d-none");
                analyzeBtn.disabled = false;
            }, 3000);
        });
    }
});
