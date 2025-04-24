// Initialize dark mode based on user preference
document.addEventListener('DOMContentLoaded', function () {
    // Check if dark mode is stored in localStorage
    const darkMode = localStorage.getItem('darkMode') === 'true';

    if (darkMode) {
        document.body.classList.add('dark-mode');
    }

    // Listen for dark mode toggle from the switch
    const observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.attributeName === 'class') {
                const darkModeEnabled = document.body.classList.contains('dark-mode');
                localStorage.setItem('darkMode', darkModeEnabled);
            }
        });
    });

    observer.observe(document.body, { attributes: true });

    // Apply custom styles to Plotly graphs 
    const updatePlotlyTheme = function () {
        const isDarkMode = document.body.classList.contains('dark-mode');

        // Find all plotly graphs and apply theme
        const graphs = document.querySelectorAll('.js-plotly-plot');
        graphs.forEach(function (graph) {
            if (graph._fullLayout) {
                const update = {
                    'paper_bgcolor': isDarkMode ? '#222' : '#fff',
                    'plot_bgcolor': isDarkMode ? '#333' : '#fff',
                    'font.color': isDarkMode ? '#ddd' : '#333',
                };
                Plotly.relayout(graph, update);
            }
        });
    };

    // Watch for theme changes to update graphs
    const themeObserver = new MutationObserver(function (mutations) {
        updatePlotlyTheme();
    });

    themeObserver.observe(document.body, { attributes: true, attributeFilter: ['class'] });

    // Update graphs when they're created or updated
    document.addEventListener('plotly_afterplot', updatePlotlyTheme);
});

// Apply responsive design to graphs
window.addEventListener('resize', function () {
    // Resize all Plotly graphs to fit container
    const graphs = document.querySelectorAll('.js-plotly-plot');
    graphs.forEach(function (graph) {
        Plotly.Plots.resize(graph);
    });
});