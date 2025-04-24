# config.py
import os

# Database settings
DB_PATH = os.environ.get('POWER_DATA_DB_PATH', 'power_data.db')

# Analysis settings
DEFAULT_SAMPLE_LIMIT = 10000
MAX_SAMPLE_LIMIT = 1000000  # 1M

# Slider settings
SAMPLE_LIMIT_STEPS = [10000, 50000, 100000, 500000, 1000000, 0]  # 0 (All) is at the end
SAMPLE_LIMIT_MARKS = {
    10000: '10k',
    50000: '50k',
    100000: '100k',
    500000: '500k',
    1000000: '1M',
    0: 'All'  # All comes after 1M
}

# UI settings
APP_TITLE = 'WaveformPro'
APP_SUBTITLE = 'Advanced Power Quality Analysis System'

# File handling
ALLOWED_EXTENSIONS = ['db', 'csv', 'xlsx', 'xls', 'bin']
UPLOAD_FOLDER = 'uploads'

# File format options
FILE_FORMAT_OPTIONS = [
    {'label': 'SQLite', 'value': 'sqlite'},
    {'label': 'CSV', 'value': 'csv'},
    {'label': 'Excel', 'value': 'excel'},
    {'label': 'Binary', 'value': 'binary'}
]
DEFAULT_FILE_FORMAT = 'sqlite'

# FFT settings
DEFAULT_WINDOW_TYPE = 'hann'
DEFAULT_FFT_SCALE = 'db'
DEFAULT_MAX_FREQ = 500

# Harmonics settings
DEFAULT_HARMONICS_TO_SHOW = 15
DEFAULT_FUNDAMENTAL_FREQ = 60
HARMONICS_VIEW_OPTIONS = [
    {'label': 'Harmonic Spectrum', 'value': 'spectrum'},
    {'label': 'Reconstruction', 'value': 'reconstruction'},
    {'label': 'Components', 'value': 'components'}
]

# Power quality settings
FLICKER_PST_THRESHOLD = 1.0
FLICKER_PLT_THRESHOLD = 0.8

# Theme settings
DARK_MODE_STYLES = {
    'background': '#1e1e1e',
    'text': '#ffffff',
    'card_bg': '#2d2d2d',
    'card_header': '#3d3d3d',
    'plot_bg': '#2d2d2d',
    'plot_paper_bg': '#1e1e1e',
    'plot_grid': '#444444',
    'header_bg': '#0066cc',
    'tab_active': '#0066cc',
    'tab_inactive': '#555555',
    'button_primary': '#0066cc',
    'button_secondary': '#555555',
    'dropdown_bg': '#3d3d3d',
    'input_bg': '#3d3d3d',
    'input_text': '#ffffff',
    'table_header_bg': '#3d3d3d',
    'table_row_even': '#2d2d2d',
    'table_row_odd': '#353535',
    'slider_rail': '#555555',
    'slider_track': '#0066cc',
    'slider_handle': '#0066cc'
}

LIGHT_MODE_STYLES = {
    'background': '#ffffff',
    'text': '#333333',
    'card_bg': '#ffffff',
    'card_header': '#f8f9fa',
    'plot_bg': '#ffffff',
    'plot_paper_bg': '#ffffff',
    'plot_grid': '#eeeeee',
    'header_bg': '#0066cc',
    'tab_active': '#0066cc',
    'tab_inactive': '#dddddd',
    'button_primary': '#0066cc',
    'button_secondary': '#6c757d',
    'dropdown_bg': '#ffffff',
    'input_bg': '#ffffff',
    'input_text': '#333333',
    'table_header_bg': '#f8f9fa',
    'table_row_even': '#ffffff',
    'table_row_odd': '#f8f9fa',
    'slider_rail': '#dddddd',
    'slider_track': '#0066cc',
    'slider_handle': '#0066cc'
}

# Plotting settings
TIME_DOMAIN_LINE_COLOR = '#0066cc'
FFT_LINE_COLOR = '#0066cc'
MEAN_LINE_COLOR = '#ff0000'
RMS_LINE_COLOR = '#00cc00'
ENVELOPE_COLOR = '#9933cc'