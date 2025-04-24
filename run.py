# run.py
import os
import sys
import logging
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Make sure Python can find the modules relative to this file
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('waveform_pro.log')
    ]
)
logger = logging.getLogger(__name__)

# Create the Dash app directly in run.py
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)

# Set app title
app.title = "WaveformPro"

# Import your layout function
try:
    from app.layout.main_layout import create_main_layout
    app.layout = create_main_layout()
except ImportError:
    # Fallback to a simple layout if the import fails
    app.layout = html.Div([
        html.H1("WaveformPro"),
        html.P("There was an error loading the application layout."),
        html.Pre(id='error-display')
    ])

# Register callbacks
try:
    # Try to import and register all callbacks
    from app.callbacks import register_all_callbacks
    register_all_callbacks(app)
except ImportError as e:
    logger.error(f"Error importing callbacks: {e}")
    # Continue without callbacks for now

def main():
    """Main function to run the WaveformPro application"""
    try:
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 8050))
        
        logger.info(f"Starting WaveformPro on port {port}")
        
        # Run the app
        app.run_server(debug=True, port=port)
        
    except Exception as e:
        logger.error(f"Error starting application: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()