# Waveform Pro - Power Data Analysis Tool

## Supported Data Formats

This tool supports the following data formats:

### CSV Files
- Standard comma-separated values format
- Expected columns: time, current
- Can include metadata in header rows

### Excel Files (.xlsx, .xls)
- Data sheet with time and current columns
- Optional metadata sheet

### JSON Files
- Standard JSON format with time and current arrays
- Can include metadata section

### HDF5 Files (.h5, .hdf5)
- Scientific data format commonly used for large datasets
- Expected datasets: time, current
- Can include metadata attributes

### COMTRADE Files (.cfg, .dat)
- Common format for Transient Data Exchange
- Used by power utilities and protection systems
- Supports both ASCII and binary formats

## Usage

### Importing Data


Future Enhancements
1. Fix Dark Mode Styling
We need to ensure proper contrast for all text elements in dark mode. This requires updating the CSS to set appropriate text colors for dark backgrounds.
2. Add File Selection Interface
Let's implement a file selection component that allows users to:

Browse and select individual files
Display the selected file details
Import the selected file

3. Additional Functionality Improvements
Data Management Features

Data Browser: Add a dedicated panel to browse all imported data files
Data Filtering: Allow filtering data by time range or specific conditions
Batch Processing: Enable processing multiple files with the same settings
Data Comparison: Add ability to overlay/compare data from different files
Data Snapshots: Create and save snapshots of specific analysis states
Data Export Options: Expand export formats to include PNG/SVG plots and PDF reports

Analysis Enhancements

Custom Filters: Allow users to design custom filters with visual feedback
Anomaly Detection: Add automatic detection of abnormal patterns
Machine Learning Integration: Add predictive analytics for power quality
Statistical Analysis: Add more advanced statistical tools for signal analysis
Correlation Analysis: Add tools to find correlations between different phases

Visualization Improvements

Interactive Plots: Make graphs fully interactive with zoom, pan, and selection
3D Visualization: Add 3D plots for multi-variable relationships
Real-time Monitoring: Add capability to monitor data sources in real-time
Customizable Dashboards: Allow users to create custom dashboard layouts
Annotation Tools: Add ability to annotate and mark important features on graphs

User Experience

User Preferences: Add a preferences panel to customize application behavior
Color Schemes: Allow selection of different color schemes beyond dark/light
Keyboard Shortcuts: Implement keyboard shortcuts for common operations
Tutorial System: Add an interactive tutorial for new users
Search Functionality: Add ability to search within the dataset
