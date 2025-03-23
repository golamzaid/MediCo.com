// Custom JavaScript for Brain Tumor Detection app

// Add any custom JavaScript functionality here
console.log('Brain Tumor Detection app loaded');

// Example: Add active class to current nav item
document.addEventListener('DOMContentLoaded', function() {
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        if (currentLocation === linkPath) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });

    // Example: Smooth scroll for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop,
                    behavior: 'smooth'
                });
            }
        });
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

    // Example: Show a confirmation dialog before form submission
    const forms = document.querySelectorAll('form.confirmation-required');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const confirmed = confirm('Are you sure you want to submit this form?');
            if (!confirmed) {
                e.preventDefault();
            }
        });
    });

    // Example: Toggle visibility of elements
    const toggleButtons = document.querySelectorAll('[data-toggle="visibility"]');
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                targetElement.classList.toggle('hidden');
            }
        });
    });

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the default form submission

        analyzeText.classList.add('d-none');
        loadingText.classList.remove('d-none');
        analyzeBtn.disabled = true;

        // Simulate the analysis process (replace this with an actual AJAX request if needed)
        setTimeout(() => {
            // Example: Simulate receiving the image path from the server
            const analyzedImagePath = imagePreview.src; // Use the same image for demonstration

            // Update the results section
            const resultsContainer = document.getElementById('results-container');
            const resultImage = document.getElementById('result-image');
            resultImage.src = analyzedImagePath; // Set the analyzed image path
            resultsContainer.classList.remove('d-none'); // Show the results section

            // Hide the loading spinner
            loadingText.classList.add('d-none');
            analyzeText.classList.remove('d-none');
            analyzeBtn.disabled = false;
        }, 3000); // Simulate a 3-second analysis delay
    });
});