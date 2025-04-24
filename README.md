# WaveformPro - Advanced Power Data Analysis Tool

WaveformPro is a comprehensive data analysis and visualization platform designed for power quality analysis, current waveform examination, and harmonic studies.

## Features

- **Multi-format Data Import**: Supports CSV, Excel, SQLite, JSON, HDF5, COMTRADE, and binary formats
- **Interactive Data Visualization**: Explore data with responsive, interactive plots
- **Comprehensive Analysis Tools**:
  - Time domain analysis
  - Frequency domain (FFT)
  - Harmonic analysis with THD calculation
  - Interharmonics analysis
  - Power quality assessment
  - Transient detection
  - Wavelet decomposition
  - Short-time Fourier Transform (STFT)
  - Cepstrum analysis
  - Waveform distortion metrics
  - Multi-phase comparison

- **Advanced Visualization**: Multiple visualization options with customizable parameters
- **Report Generation**: Create comprehensive reports with selected analyses

## Supported Data Formats

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

### SQLite Database
- Tables containing time and current values
- Support for different measurement setups

## Usage Guide

### Data Selection Tab

1. **Database Data**:
   - Select a system and phase from the dropdown menus
   - Adjust the sample limit slider to control the amount of data loaded
   - Click "Load Data" to load the selected data

2. **Upload Data**:
   - Select the file format from the radio buttons
   - Drag and drop a file or click to browse and select
   - The app will automatically parse and display the data

### Analysis Tab

1. Select an analysis type from the options on the left
2. Adjust the analysis parameters as needed
3. Click "Run Analysis" to perform the selected analysis
4. View the results displayed on the right

### Visualization Tab

1. Select a visualization type from the options on the left
2. Adjust visualization parameters as needed
3. Click "Update Visualization" to generate the visual
4. Use the export options to save as PNG, SVG, or CSV

### Reports Tab

1. Enter a title for your report
2. Select which analyses to include in the report
3. Click "Generate Report" to create a comprehensive report
4. The report preview will display on the right

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/waveform-pro.git
cd waveform-pro

2. Install dependencies:
pip install -r requirements.txt

3. Run the application:
python run.py

## Future Enhancements

- Machine Learning integration for anomaly detection
- Real-time monitoring capabilities
- Custom filter design tools
- Additional data export formats
- 3D visualization for multi-variable analysis

