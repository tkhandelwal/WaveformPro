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
                updateAllThemes();
            }
        });
    });

    observer.observe(document.body, { attributes: true });

    // Function to update all themed elements
    function updateAllThemes() {
        updatePlotlyTheme();
        updateDomElements();
    }

    // Update Plotly graphs to match theme
    function updatePlotlyTheme() {
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

                // Update axis colors
                if (graph._fullLayout.xaxis) {
                    update['xaxis.gridcolor'] = isDarkMode ? '#444' : '#eee';
                    update['xaxis.linecolor'] = isDarkMode ? '#444' : '#eee';
                    update['xaxis.tickfont.color'] = isDarkMode ? '#ddd' : '#333';
                    update['xaxis.title.font.color'] = isDarkMode ? '#ddd' : '#333';
                }

                if (graph._fullLayout.yaxis) {
                    update['yaxis.gridcolor'] = isDarkMode ? '#444' : '#eee';
                    update['yaxis.linecolor'] = isDarkMode ? '#444' : '#eee';
                    update['yaxis.tickfont.color'] = isDarkMode ? '#ddd' : '#333';
                    update['yaxis.title.font.color'] = isDarkMode ? '#ddd' : '#333';
                }

                // Update legend colors
                if (graph._fullLayout.legend) {
                    update['legend.font.color'] = isDarkMode ? '#ddd' : '#333';
                    update['legend.bgcolor'] = isDarkMode ? '#333' : '#fff';
                    update['legend.bordercolor'] = isDarkMode ? '#444' : '#e2e2e2';
                }

                try {
                    Plotly.relayout(graph, update);
                } catch (e) {
                    console.warn('Failed to update graph theme:', e);
                }
            }
        });
    }

    // Update DOM elements that might need theme adjustments
    function updateDomElements() {
        const isDarkMode = document.body.classList.contains('dark-mode');

        // Update modebar buttons
        const modebarButtons = document.querySelectorAll('.modebar-btn path');
        modebarButtons.forEach(function (path) {
            path.setAttribute('fill', isDarkMode ? '#fff' : '#666');
        });

        // Update other themed elements as needed
    }

    // Update whenever the page content changes
    const contentObserver = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.addedNodes.length > 0) {
                // Check if any Plotly graphs were added
                let needsUpdate = false;
                mutation.addedNodes.forEach(function (node) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (node.classList && node.classList.contains('js-plotly-plot')) {
                            needsUpdate = true;
                        } else if (node.querySelector && node.querySelector('.js-plotly-plot')) {
                            needsUpdate = true;
                        }
                    }
                });

                if (needsUpdate) {
                    // Wait a bit for Plotly to initialize
                    setTimeout(updatePlotlyTheme, 100);
                }
            }
        });
    });

    contentObserver.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Update graphs when they're created or updated
    document.addEventListener('plotly_afterplot', updatePlotlyTheme);

    // Initial update
    setTimeout(updateAllThemes, 100);
});

// Apply responsive design to graphs
window.addEventListener('resize', function () {
    // Resize all Plotly graphs to fit container
    const graphs = document.querySelectorAll('.js-plotly-plot');
    graphs.forEach(function (graph) {
        try {
            Plotly.Plots.resize(graph);
        } catch (e) {
            console.warn('Failed to resize graph:', e);
        }
    });
});

// Add helper functions for dynamic theme changes
window.toggleDarkMode = function (enable) {
    if (enable) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
};