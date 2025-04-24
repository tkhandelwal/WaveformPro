# layouts.py
from dash import dcc, html
import dash_bootstrap_components as dbc
from config import (
    APP_TITLE, APP_SUBTITLE, DEFAULT_SAMPLE_LIMIT, MAX_SAMPLE_LIMIT,
    FILE_FORMAT_OPTIONS, DEFAULT_FILE_FORMAT, SAMPLE_LIMIT_MARKS, SAMPLE_LIMIT_STEPS,
    DARK_MODE_STYLES, LIGHT_MODE_STYLES, HARMONICS_VIEW_OPTIONS
)

def create_header(dark_mode=False):
    """Create application header"""
    header_style = {
        'backgroundColor': DARK_MODE_STYLES['header_bg'] if dark_mode else LIGHT_MODE_STYLES['header_bg'],
        'padding': '10px 20px',
        'color': '#ffffff',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    }
    
    return html.Div([
        html.Div([
            html.H1(APP_TITLE, style={'margin': '0', 'fontSize': '24px'}),
            html.Span(APP_SUBTITLE, style={'marginLeft': '15px', 'fontSize': '14px'})
        ], style={'display': 'flex', 'alignItems': 'center'}),
        
        html.Div([
            html.Label('Dark Mode', 
                      style={'marginRight': '10px', 'color': '#ffffff'}),
            dbc.Switch(
                id='dark-mode-switch',
                value=dark_mode,
                className="ml-auto"
            )
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style=header_style)

def create_tabs(dark_mode=False):
    """Create main navigation tabs"""
    tab_style = {
        'padding': '10px 15px',
        'borderRadius': '5px 5px 0 0',
        'backgroundColor': DARK_MODE_STYLES['tab_inactive'] if dark_mode else LIGHT_MODE_STYLES['tab_inactive'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'marginRight': '2px',
        'cursor': 'pointer'
    }
    
    tab_selected_style = {
        'padding': '10px 15px',
        'borderRadius': '5px 5px 0 0',
        'backgroundColor': DARK_MODE_STYLES['tab_active'] if dark_mode else LIGHT_MODE_STYLES['tab_active'],
        'color': '#ffffff',
        'marginRight': '2px',
        'fontWeight': 'bold'
    }
    
    return dcc.Tabs(
        id='main-tabs',
        value='data-selection',
        children=[
            dcc.Tab(label='Data Selection', value='data-selection', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Analysis', value='analysis', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Visualization', value='visualization', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Reports', value='reports', style=tab_style, selected_style=tab_selected_style),
        ],
        style={
            'marginBottom': '20px',
            'borderBottom': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}'
        }
    )

def create_data_selection_content(systems, dark_mode=False):
    """Create data selection tab content"""
    system_options = [{'label': s, 'value': s} for s in systems]
    
    card_style = {
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': 'none',
        'borderRadius': '5px',
        'padding': '20px',
        'marginBottom': '20px'
    }
    
    dropdown_style = {
        'backgroundColor': DARK_MODE_STYLES['dropdown_bg'] if dark_mode else LIGHT_MODE_STYLES['dropdown_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}',
        'borderRadius': '5px'
    }
    
    button_style = {
        'backgroundColor': DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
        'color': '#ffffff',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
    
    return html.Div([
        # Database and Upload sections
        dbc.Row([
            # Database data selection
            dbc.Col([
                html.Div([
                    html.H4("Database Data", style={'marginBottom': '20px'}),
                    
                    # System and Phase selection
                    dbc.Row([
                        dbc.Col([
                            html.Label("System:", style={'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='system-dropdown',
                                options=system_options,
                                value=systems[0] if systems else None,
                                clearable=False,
                                style=dropdown_style
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Phase:", style={'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='phase-dropdown', 
                                clearable=False,
                                style=dropdown_style
                            )
                        ], width=6),
                    ], className="mb-3"),
                    
                    # Sample limit slider - fixed to show All at the beginning
                    html.Label("Sample Limit:", style={'marginBottom': '5px'}),
                    dcc.Slider(
                        id='sample-limit-slider',
                        min=min(SAMPLE_LIMIT_STEPS),
                        max=max(SAMPLE_LIMIT_STEPS),
                        step=None,
                        value=DEFAULT_SAMPLE_LIMIT,
                        marks=SAMPLE_LIMIT_MARKS
                    ),
                    
                    # Load button
                    html.Button(
                        "Load Data", 
                        id="load-button", 
                        style={**button_style, 'marginTop': '20px'}
                    ),
                ], style=card_style)
            ], width=6),
            
            # File upload
            dbc.Col([
                html.Div([
                    html.H4("Upload Data", style={'marginBottom': '20px'}),
                    
                    # Upload component
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files', style={'color': DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary']})
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '100px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'marginBottom': '20px',
                            'backgroundColor': DARK_MODE_STYLES['input_bg'] if dark_mode else LIGHT_MODE_STYLES['input_bg'],
                            'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text']
                        },
                        className='upload-box',
                        multiple=False
                    ),
                    
                    # File format selection
                    html.Label("File Format:", style={'marginBottom': '10px'}),
                    dbc.RadioItems(
                        id='file-format',
                        options=FILE_FORMAT_OPTIONS,
                        value=DEFAULT_FILE_FORMAT,
                        inline=True,
                        labelStyle={'marginRight': '15px'}
                    ),
                    
                    # Upload status area
                    html.Div(id='upload-status', style={'marginTop': '10px'})
                ], style=card_style)
            ], width=6),
        ]),
        
        # Data Summary and Preview
        html.Div([
            html.Div(id='data-summary', className="data-summary-container"),
            html.Div(id='data-preview-plot', className="data-preview-container")
        ], style=card_style)
    ])

def create_analysis_content(dark_mode=False):
    """Create analysis tab content"""
    card_style = {
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': 'none',
        'borderRadius': '5px',
        'padding': '20px',
        'marginBottom': '20px'
    }
    
    button_style = {
        'backgroundColor': DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
        'color': '#ffffff',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
    
    return html.Div([
        dbc.Row([
            # Analysis Options
            dbc.Col([
                html.Div([
                    html.H4("Analysis Options", style={'marginBottom': '20px'}),
                    
                    # Analysis Type
                    html.Label("Analysis Type:", style={'marginBottom': '10px'}),
                    dbc.RadioItems(
                        id='analysis-type',
                        options=[
                            {'label': 'Time Domain', 'value': 'time'},
                            {'label': 'Frequency Domain (FFT)', 'value': 'fft'},
                            {'label': 'Harmonics', 'value': 'harmonics'},
                            {'label': 'Interharmonics', 'value': 'interharmonics'},
                            {'label': 'Power Quality', 'value': 'power_quality'},
                            {'label': 'Transients', 'value': 'transients'},
                            {'label': 'Wavelet', 'value': 'wavelet'},
                            {'label': 'STFT', 'value': 'stft'},
                            {'label': 'Cepstrum', 'value': 'cepstrum'},
                            {'label': 'Waveform Distortion', 'value': 'distortion'},
                            {'label': 'Multi-Phase', 'value': 'multi_phase'}
                        ],
                        value='time',
                        labelStyle={'display': 'block', 'marginBottom': '10px'}
                    ),
                    
                    # Analysis Parameters - dynamically updated by callback
                    html.Div(id='analysis-parameters', className="mt-4"),
                    
                    # Run Analysis Button
                    html.Button(
                        "Run Analysis", 
                        id="run-analysis-button", 
                        style={**button_style, 'marginTop': '20px'}
                    ),
                ], style=card_style)
            ], width=4),
            
            # Analysis Results
            dbc.Col([
                html.Div([
                    dbc.Spinner(
                        html.Div(id='analysis-results'),
                        color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
                        type="border",
                    )
                ], style=card_style)
            ], width=8),
        ]),
    ])

def create_visualization_content(dark_mode=False):
    """Create visualization tab content"""
    card_style = {
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': 'none',
        'borderRadius': '5px',
        'padding': '20px',
        'marginBottom': '20px'
    }
    
    button_style = {
        'backgroundColor': DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
        'color': '#ffffff',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
    
    export_button_style = {
        'backgroundColor': DARK_MODE_STYLES['button_secondary'] if dark_mode else LIGHT_MODE_STYLES['button_secondary'],
        'color': '#ffffff',
        'border': 'none',
        'padding': '5px 15px',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginTop': '10px'
    }
    
    dropdown_style = {
        'backgroundColor': DARK_MODE_STYLES['dropdown_bg'] if dark_mode else LIGHT_MODE_STYLES['dropdown_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}',
        'borderRadius': '5px'
    }
    
    return html.Div([
        dbc.Row([
            # Visualization Options
            dbc.Col([
                html.Div([
                    html.H4("Visualization Options", style={'marginBottom': '20px'}),
                    
                    # Visualization Type
                    html.Label("Visualization Type:", style={'marginBottom': '10px'}),
                    dbc.RadioItems(
                        id='viz-type',
                        options=[
                            {'label': 'Time Series', 'value': 'time'},
                            {'label': 'FFT Plot', 'value': 'fft'},
                            {'label': 'STFT', 'value': 'stft'},
                            {'label': 'Spectrogram', 'value': 'spectrogram'},
                            {'label': 'Harmonic Spectrum', 'value': 'harmonics'},
                            {'label': 'Wavelet Decomposition', 'value': 'wavelet'},
                            {'label': 'Transient Analysis', 'value': 'transients'},
                            {'label': 'Multi-Phase', 'value': 'multi_phase'}
                            {'label': '3D Harmonic Visualization', 'value': '3d_harmonics'}  # Add this line

                        ],
                        value='time',
                        labelStyle={'display': 'block', 'marginBottom': '10px'}
                    ),
                    
                    # Visualization Parameters - dynamically updated by callback
                    html.Div(id='viz-parameters', className="mt-4"),
                    
                    # Update Visualization Button
                    html.Button(
                        "Update Visualization", 
                        id="update-viz-button", 
                        style={**button_style, 'marginTop': '20px'}
                    ),
                    
                    # Export Options
                    html.Div([
                        html.H5("Export", style={'marginTop': '30px', 'marginBottom': '10px'}),
                        dbc.RadioItems(
                            id='export-format',
                            options=[
                                {'label': 'PNG', 'value': 'png'},
                                {'label': 'SVG', 'value': 'svg'},
                                {'label': 'CSV (Data)', 'value': 'csv'}
                            ],
                            value='png',
                            inline=True,
                            labelStyle={'marginRight': '10px'}
                        ),
                        html.Button(
                            "Export", 
                            id="export-button", 
                            style=export_button_style
                        )
                    ]),
                ], style=card_style)
            ], width=3),
            
            # Visualization Area
            dbc.Col([
                html.Div([
                    dbc.Spinner(
                        html.Div(id='visualization-area'),
                        color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
                        type="border",
                    )
                ], style=card_style)
            ], width=9),
        ]),
    ])

def create_reports_content(dark_mode=False):
    """Create reports tab content"""
    card_style = {
        'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'border': 'none',
        'borderRadius': '5px',
        'padding': '20px',
        'marginBottom': '20px'
    }
    
    input_style = {
        'backgroundColor': DARK_MODE_STYLES['input_bg'] if dark_mode else LIGHT_MODE_STYLES['input_bg'],
        'color': DARK_MODE_STYLES['input_text'] if dark_mode else LIGHT_MODE_STYLES['input_text'],
        'border': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}',
        'borderRadius': '5px',
        'padding': '8px 12px',
        'width': '100%',
        'marginBottom': '20px'
    }
    
    button_style = {
        'backgroundColor': DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
        'color': '#ffffff',
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer'
    }
    
    preview_style = {
        'backgroundColor': DARK_MODE_STYLES['input_bg'] if dark_mode else LIGHT_MODE_STYLES['input_bg'],
        'border': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}',
        'borderRadius': '5px',
        'padding': '15px',
        'height': '80vh',
        'overflow': 'auto'
    }
    
    return html.Div([
        dbc.Row([
            # Report Options
            dbc.Col([
                html.Div([
                    html.H4("Report Generation", style={'marginBottom': '20px'}),
                    
                    # Report Title
                    html.Label("Report Title:", style={'marginBottom': '5px'}),
                    dbc.Input(
                        id="report-title", 
                        type="text", 
                        placeholder="Enter report title",
                        style=input_style
                    ),
                    
                    # Included Analysis
                    html.Label("Included Analysis:", style={'marginBottom': '10px'}),
                    dbc.Checklist(
                        id="report-include",
                        options=[
                            {"label": "Time Domain", "value": "time"},
                            {"label": "Frequency Analysis", "value": "fft"},
                            {"label": "Harmonics", "value": "harmonics"},
                            {"label": "Power Quality", "value": "power_quality"},
                            {"label": "Transients", "value": "transients"},
                            {"label": "Wavelet Analysis", "value": "wavelet"},
                            {"label": "Multi-Phase Comparison", "value": "multi_phase"}
                        ],
                        value=["time", "fft", "harmonics"],
                        labelStyle={'display': 'block', 'marginBottom': '8px'}
                    ),
                    
                    # Generate Report Button
                    html.Button(
                        "Generate Report", 
                        id="generate-report-button", 
                        style={**button_style, 'marginTop': '20px'}
                    ),
                ], style=card_style)
            ], width=4),
            
            # Report Preview
            dbc.Col([
                html.Div([
                    html.H4("Report Preview", style={'marginBottom': '15px'}),
                    html.Div(id="report-preview", style=preview_style)
                ], style=card_style)
            ], width=8),
        ]),
    ])

def create_main_layout(systems, dark_mode=False):
    """Create the main application layout"""
    background_style = {
        'backgroundColor': DARK_MODE_STYLES['background'] if dark_mode else LIGHT_MODE_STYLES['background'],
        'color': DARK_MODE_STYLES['text'] if dark_mode else LIGHT_MODE_STYLES['text'],
        'minHeight': '100vh'
    }
    
    status_style = {
        'padding': '10px 20px',
        'backgroundColor': '#d4edda' if not dark_mode else '#1e3927',
        'color': '#155724' if not dark_mode else '#ffffff',
        'borderRadius': '5px',
        'marginTop': '20px',
        'marginBottom': '20px',
        'opacity': '0'  # Initially hidden
    }
    
    return html.Div([
        # Theme store for tracking dark/light mode
        dcc.Store(id='theme-store', data={'dark_mode': dark_mode}),
        
        # Application header
        create_header(dark_mode),
        
        # Main container
        html.Div([
            # Tabs navigation
            create_tabs(dark_mode),
            
            # Tab content - controlled by callback
            html.Div(id='tab-content', children=[
                # Default tab is data selection
                create_data_selection_content(systems, dark_mode)
            ]),
            
            # Status area
            html.Div(id='status-area', style=status_style),
            
            # Hidden store components for sharing data between callbacks
            dcc.Store(id='current-data'),
            dcc.Store(id='analysis-results-store'),
            dcc.Store(id='visualization-store'),
            dcc.Store(id='harmonics-data'),
            dcc.Store(id='wavelet-data'),
            dcc.Store(id='power-quality-data'),
            dcc.Store(id='transient-data'),
            dcc.Store(id='multi-phase-data'),
            dcc.Store(id='stft-data'),
            dcc.Store(id='interharmonics-data'),
            dcc.Store(id='cepstrum-data'),
            dcc.Store(id='distortion-data'),
            
            # Hidden components for analysis parameters
            html.Div([
                # For Time Domain analysis
                dcc.Store(id='time-options', data=['show_mean', 'show_rms']),
                
                # For FFT analysis
                dcc.Store(id='fft-window-type', data='hann'),
                dcc.Store(id='fft-scale', data='db'),
                dcc.Store(id='fft-max-freq', data=500),
                
                # For Harmonics analysis
                dcc.Store(id='harmonics-count', data=15),
                dcc.Store(id='fundamental-freq', data=60),
                dcc.Store(id='harmonics-view', data='spectrum'),
                dcc.Store(id='harmonics-selection', data=[1, 2, 3, 4, 5, 6, 7]),
                
                # For Power Quality analysis
                dcc.Store(id='power-quality-type', data='flicker'),
                dcc.Store(id='power-quality-sensitivity', data=2),
                
                # For Transient analysis
                dcc.Store(id='transient-sensitivity', data=2),
                dcc.Store(id='transient-window', data=20),
                
                # For Wavelet analysis
                dcc.Store(id='wavelet-type', data='db4'),
                dcc.Store(id='wavelet-level', data=2),
                
                # For STFT analysis
                dcc.Store(id='stft-window-size', data=256),
                dcc.Store(id='stft-overlap', data=50),
                dcc.Store(id='stft-window-type', data='hann'),
                dcc.Store(id='stft-colormap', data='viridis'),
                
                # For Interharmonics analysis
                dcc.Store(id='interharmonics-fund-freq', data=60),
                dcc.Store(id='interharmonics-groups', data=10),
                
                # For Cepstrum analysis
                dcc.Store(id='cepstrum-window-type', data='hann'),
                dcc.Store(id='cepstrum-threshold', data=10),
                
                # For Distortion analysis
                dcc.Store(id='distortion-fund-freq', data=60),
                dcc.Store(id='distortion-harmonics', data=40),
                dcc.Store(id='distortion-options', data=['show_even', 'show_odd', 'show_triplen']),
                
                # For Multi-phase analysis
                dcc.Store(id='multi-phase-selection', data=['1', '2', '3']),
                dcc.Store(id='multi-phase-type', data='thd'),
                dcc.Store(id='multi-phase-plot-style', data='overlay')
            ], style={'display': 'none'})
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '20px'
        }),
    ], style=background_style, id='app-container')