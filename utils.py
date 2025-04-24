import numpy as np
import json
from dash import html, dcc
import dash_bootstrap_components as dbc
import logging
import plotly.graph_objs as go
from config import DARK_MODE_STYLES, LIGHT_MODE_STYLES

# Configure logging
def setup_logging(level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('waveform_pro.log')
        ]
    )
    return logging.getLogger(__name__)

# JSON conversion functions
def convert_to_json_serializable(data):
    """Convert numpy arrays and other non-serializable objects to JSON serializable format"""
    if data is None:
        return None
        
    json_data = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        elif isinstance(value, np.int64) or isinstance(value, np.int32):
            json_data[key] = int(value)
        elif isinstance(value, np.float64) or isinstance(value, np.float32):
            json_data[key] = float(value)
        else:
            json_data[key] = value
    
    return json_data

def convert_from_json(json_data):
    """Convert JSON data back to numpy arrays"""
    if json_data is None:
        return None
        
    data = {}
    
    for key, value in json_data.items():
        if isinstance(value, list):
            data[key] = np.array(value)
        else:
            data[key] = value
    
    return data

# UI Helper functions
def create_status_message(message, status_type='info', dark_mode=False):
    """Create a status message with appropriate styling"""
    styles = {
        'success': {
            'backgroundColor': '#d4edda' if not dark_mode else '#1e3927',
            'color': '#155724' if not dark_mode else '#ffffff',
            'borderColor': '#c3e6cb' if not dark_mode else '#2a5738'
        },
        'warning': {
            'backgroundColor': '#fff3cd' if not dark_mode else '#433816',
            'color': '#856404' if not dark_mode else '#ffffff',
            'borderColor': '#ffeeba' if not dark_mode else '#5c4f1d'
        },
        'danger': {
            'backgroundColor': '#f8d7da' if not dark_mode else '#3e1c1f',
            'color': '#721c24' if not dark_mode else '#ffffff',
            'borderColor': '#f5c6cb' if not dark_mode else '#542529'
        },
        'info': {
            'backgroundColor': '#d1ecf1' if not dark_mode else '#1a3a3e',
            'color': '#0c5460' if not dark_mode else '#ffffff',
            'borderColor': '#bee5eb' if not dark_mode else '#265459'
        }
    }
    
    style = styles.get(status_type, styles['info'])
    style.update({
        'padding': '10px 20px',
        'borderRadius': '5px',
        'marginTop': '20px',
        'marginBottom': '20px',
        'opacity': '1'
    })
    
    return html.Div(message, style=style)

def create_data_summary(data, source_info, dark_mode=False):
    """Create a summary card for data"""
    if data is None or 'current' not in data or len(data['current']) == 0:
        return html.Div("No data available")
    
    try:
        # Calculate basic metrics
        num_samples = len(data['current'])
        duration = data['time'][-1] - data['time'][0] if len(data['time']) > 0 else 0
        sample_rate = 1 / data['sample_interval']
        mean_value = np.mean(data['current'])
        rms_value = np.sqrt(np.mean(np.square(data['current'])))
        
        # Apply theme styles
        card_style = {
            'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
            'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
            'border': 'none',
            'borderRadius': '5px',
            'padding': '20px',
            'marginBottom': '20px'
        }
        
        header_style = {
            'backgroundColor': DARK_MODE_STYLES['card_header'] if dark_mode else LIGHT_MODE_STYLES['card_header'],
            'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
            'padding': '10px 20px',
            'borderTopLeftRadius': '5px',
            'borderTopRightRadius': '5px'
        }
        
        # Style for the summary items
        item_style = {
            'margin': '10px 0',
            'fontSize': '14px'
        }
        
        return html.Div([
            # Header
            html.Div("Data Summary", style=header_style),
            
            # Summary content
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div(f"Source: {source_info}", style=item_style),
                        html.Div(f"Number of samples: {num_samples:,}", style=item_style),
                        html.Div(f"Duration: {duration:.4f} seconds", style=item_style),
                    ], width=6),
                    dbc.Col([
                        html.Div(f"Sampling rate: {sample_rate:.2f} Hz", style=item_style),
                        html.Div(f"Mean value: {mean_value:.4f} A", style=item_style),
                        html.Div(f"RMS value: {rms_value:.4f} A", style=item_style),
                    ], width=6),
                ])
            ], style={'padding': '10px 20px'})
        ], style=card_style)
    except Exception as e:
        logging.error(f"Error creating data summary: {e}")
        return html.Div(f"Error creating data summary: {str(e)}")

def create_preview_plot(data, dark_mode=False):
    """Create a preview plot of the data"""
    if data is None or 'current' not in data or len(data['current']) == 0:
        return html.Div("No data to preview")
    
    # Get the number of samples
    num_samples = len(data['current'])
    
    # Create the plot
    fig = go.Figure(
        data=[go.Scatter(
            x=data['time'], 
            y=data['current'],
            mode='lines',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        )],
        layout=go.Layout(
            title=f"Data Preview ({num_samples:,} points)",
            margin=dict(l=40, r=40, t=40, b=30),
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            height=400,
            template='plotly_dark' if dark_mode else 'plotly_white'
        )
    )
    
    # Apply theme
    fig = apply_theme_to_figure(fig, dark_mode)
    
    # Create card with theme styling
    card_style = {
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': 'none',
        'borderRadius': '5px',
        'padding': '0',
        'marginBottom': '20px',
        'overflow': 'hidden'
    }
    
    header_style = {
        'backgroundColor': DARK_MODE_STYLES['card_header'] if dark_mode else LIGHT_MODE_STYLES['card_header'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'padding': '10px 20px',
        'borderTopLeftRadius': '5px',
        'borderTopRightRadius': '5px'
    }
    
    return html.Div([
        # Header
        html.Div("Data Preview", style=header_style),
        
        # Plot
        html.Div([
            dcc.Graph(
                figure=fig, 
                config={
                    'displayModeBar': True,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                    'scrollZoom': True
                }
            )
        ], style={'padding': '10px'})
    ], style=card_style)

def create_metrics_table(metrics, dark_mode=False):
    """Create an HTML table from metrics data"""
    if metrics is None:
        return html.P("No metrics available.")
        
    def format_value(key, value):
        if key in ['mean', 'std', 'rms', 'peak_to_peak']:
            return f"{value:.4f} A"
        elif key in ['crest_factor', 'form_factor']:
            return f"{value:.4f}"
        elif key == 'dominant_freq':
            return f"{value:.2f} Hz"
        elif key == 'duration':
            return f"{value:.4f} s"
        else:
            return f"{value}"
    
    # Create a dict with human-readable names
    metric_names = {
        'mean': 'Mean Current',
        'median': 'Median Current',
        'std': 'Standard Deviation',
        'rms': 'RMS Current',
        'peak_to_peak': 'Peak-to-Peak',
        'crest_factor': 'Crest Factor',
        'form_factor': 'Form Factor',
        'dominant_freq': 'Dominant Frequency',
        'num_samples': 'Number of Samples',
        'duration': 'Signal Duration',
        'max_current': 'Maximum Current',
        'min_current': 'Minimum Current'
    }
    
    # Create table rows for metrics we want to display
    display_metrics = [
        'rms', 'peak_to_peak', 'crest_factor', 'form_factor',
        'mean', 'median', 'std', 'dominant_freq', 
        'max_current', 'min_current', 'num_samples', 'duration'
    ]
    
    rows = []
    for key in display_metrics:
        if key in metrics:
            rows.append(html.Tr([
                html.Td(metric_names.get(key, key), style={
                    'fontWeight': 'bold', 
                    'padding': '8px', 
                    'borderBottom': f'1px solid {DARK_MODE_STYLES["input_bg"] if dark_mode else LIGHT_MODE_STYLES["input_bg"]}'
                }),
                html.Td(format_value(key, metrics[key]), style={
                    'padding': '8px', 
                    'borderBottom': f'1px solid {DARK_MODE_STYLES["input_bg"] if dark_mode else LIGHT_MODE_STYLES["input_bg"]}'
                })
            ]))
    
    # Apply theme styles
    table_style = {
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'width': '100%',
        'borderCollapse': 'collapse'
    }
    
    header_style = {
        'backgroundColor': DARK_MODE_STYLES['card_header'] if dark_mode else LIGHT_MODE_STYLES['card_header'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'padding': '10px 20px',
        'borderTopLeftRadius': '5px',
        'borderTopRightRadius': '5px',
        'marginBottom': '15px',
        'fontWeight': 'bold'
    }
    
    return html.Div([
        html.Div("Signal Metrics", style=header_style),
        html.Table(rows, style=table_style)
    ], style={
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'borderRadius': '5px',
        'padding': '0 0 10px 0',
        'overflow': 'hidden'
    })

def apply_theme_to_figure(fig, dark_mode=False):
    """Apply theme colors to plotly figure"""
    if dark_mode:
        fig.update_layout(
            paper_bgcolor=DARK_MODE_STYLES['plot_paper_bg'],
            plot_bgcolor=DARK_MODE_STYLES['plot_bg'],
            font={'color': DARK_MODE_STYLES['text']},
            title={'font': {'color': DARK_MODE_STYLES['text']}},
            legend={'font': {'color': DARK_MODE_STYLES['text']}},
            template='plotly_dark'
        )
        fig.update_xaxes(
            gridcolor=DARK_MODE_STYLES['plot_grid'],
            zerolinecolor=DARK_MODE_STYLES['plot_grid'],
            title={'font': {'color': DARK_MODE_STYLES['text']}}
        )
        fig.update_yaxes(
            gridcolor=DARK_MODE_STYLES['plot_grid'],
            zerolinecolor=DARK_MODE_STYLES['plot_grid'],
            title={'font': {'color': DARK_MODE_STYLES['text']}}
        )
    else:
        fig.update_layout(
            paper_bgcolor=LIGHT_MODE_STYLES['plot_paper_bg'],
            plot_bgcolor=LIGHT_MODE_STYLES['plot_bg'],
            font={'color': LIGHT_MODE_STYLES['text']},
            title={'font': {'color': LIGHT_MODE_STYLES['text']}},
            legend={'font': {'color': LIGHT_MODE_STYLES['text']}},
            template='plotly_white'
        )
        fig.update_xaxes(
            gridcolor=LIGHT_MODE_STYLES['plot_grid'],
            zerolinecolor=LIGHT_MODE_STYLES['plot_grid'],
            title={'font': {'color': LIGHT_MODE_STYLES['text']}}
        )
        fig.update_yaxes(
            gridcolor=LIGHT_MODE_STYLES['plot_grid'],
            zerolinecolor=LIGHT_MODE_STYLES['plot_grid'],
            title={'font': {'color': LIGHT_MODE_STYLES['text']}}
        )
    
    return fig