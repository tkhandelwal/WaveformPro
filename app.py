import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import logging
import traceback
import json
import base64
import io
import os
import plotly.express as px

# Import project modules
from config import (
    APP_TITLE, DARK_MODE_STYLES, LIGHT_MODE_STYLES, SAMPLE_LIMIT_STEPS,
    FILE_FORMAT_OPTIONS, DEFAULT_FILE_FORMAT
)
from data_access import DataAccessLayer
from file_handlers import FileHandler
from signal_processor import SignalProcessor
from visualizers import PlotGenerator
from layouts import (
    create_main_layout, create_data_selection_content,
    create_analysis_content, create_visualization_content,
    create_reports_content
)
from utils import (
    setup_logging,
    convert_to_json_serializable,
    convert_from_json,
    create_status_message,
    create_metrics_table,
    create_data_summary,
    create_preview_plot
)

# Initialize logging
logger = setup_logging()

# Initialize the app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title=APP_TITLE
)

# Initialize server (for deployment)
server = app.server

# Initialize data access layer
data_layer = DataAccessLayer()
systems = data_layer.get_system_list()

# Initialize signal processor and plot generator
signal_processor = SignalProcessor()
plot_generator = PlotGenerator()

# Set the app layout
app.layout = create_main_layout(systems)

# ======= Callbacks =======

# Callback to handle dark mode toggle
@app.callback(
    Output('theme-store', 'data'),
    Output('app-container', 'style'),
    Output('dark-mode-switch', 'value', allow_duplicate=True),  # Allow duplicate to handle initialization
    Input('dark-mode-switch', 'value'),
    State('theme-store', 'data'),
    prevent_initial_call=True
)
def toggle_dark_mode(dark_mode_enabled, current_theme):
    if current_theme is None:
        current_theme = {'dark_mode': False}
    
    current_theme['dark_mode'] = dark_mode_enabled
    
    # Get style based on theme
    if dark_mode_enabled:
        app_style = {
            'backgroundColor': DARK_MODE_STYLES['background'],
            'color': DARK_MODE_STYLES['text'],
            'minHeight': '100vh',
            'transition': 'background-color 0.3s, color 0.3s'
        }
    else:
        app_style = {
            'backgroundColor': LIGHT_MODE_STYLES['background'],
            'color': LIGHT_MODE_STYLES['text'],
            'minHeight': '100vh',
            'transition': 'background-color 0.3s, color 0.3s'
        }
    
    # Add client-side callback to update the body class
    app.clientside_callback(
        """
        function(darkMode) {
            if(darkMode) {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('dark-mode-switch', 'id'),  # Dummy output that won't actually change
        Input('dark-mode-switch', 'value')
    )
    
    return current_theme, app_style, dark_mode_enabled

# Callback for tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    State('theme-store', 'data')
)
def render_tab_content(tab, theme_data):
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    
    if tab == 'data-selection':
        return create_data_selection_content(systems, dark_mode)
    elif tab == 'analysis':
        return create_analysis_content(dark_mode)
    elif tab == 'visualization':
        return create_visualization_content(dark_mode)
    elif tab == 'reports':
        return create_reports_content(dark_mode)
    
    # Default to data selection
    return create_data_selection_content(systems, dark_mode)

# Callback to update phase dropdown
@app.callback(
    Output('phase-dropdown', 'options'),
    Output('phase-dropdown', 'value'),
    Input('system-dropdown', 'value')
)
def update_phase_dropdown(selected_system):
    if not selected_system:
        return [], None
    
    phases = data_layer.get_phases_for_system(selected_system)
    
    # Map phase numbers to letters
    phase_map = {1: 'A', 2: 'B', 3: 'C'}
    options = [{'label': f'Phase {phase_map.get(p, p)}', 'value': p} for p in phases]
    value = phases[0] if phases else None
    
    return options, value


# Callback to load data
@app.callback(
    Output('current-data', 'data'),
    Output('data-summary', 'children'),
    Output('data-preview-plot', 'children'),
    Output('status-area', 'children'),
    Output('status-area', 'style'),
    [Input('load-button', 'n_clicks'),
     Input('upload-data', 'contents')],
    [State('system-dropdown', 'value'),
     State('phase-dropdown', 'value'),
     State('sample-limit-slider', 'value'),
     State('upload-data', 'filename'),
     State('file-format', 'value'),
     State('theme-store', 'data')]
)
def load_data(n_clicks, file_contents, system, phase, sample_limit, filename, file_format, theme_data):
    # Determine which input triggered the callback
    ctx = dash.callback_context
    trigger = ctx.triggered_id
    
    if trigger is None:
        empty_status_style = {
            'padding': '10px 20px',
            'borderRadius': '5px',
            'marginTop': '20px',
            'marginBottom': '20px',
            'opacity': '0'  # Hidden
        }
        return None, None, None, "", empty_status_style
    
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    
    if trigger == 'load-button':
        # Load from database
        if not system or phase is None:
            status_msg = create_status_message("Please select a system and phase", "warning", dark_mode)
            return None, None, None, status_msg, {'opacity': '1'}
        
        try:
            # Convert 0 to None for "All" samples
            limit = None if sample_limit == 0 else sample_limit
            
            data = data_layer.get_data_for_analysis(system, phase, limit)
            
            if data is None or len(data['current']) == 0:
                status_msg = create_status_message(
                    "No data available for the selected system and phase", "warning", dark_mode
                )
                return None, None, None, status_msg, {'opacity': '1'}
            
            # Convert data to JSON serializable format
            json_data = convert_to_json_serializable(data)
            
            # Create data summary
            phase_map = {1: 'A', 2: 'B', 3: 'C'}
            phase_letter = phase_map.get(phase, phase)
            summary = create_data_summary(data, f"System: {system}, Phase: {phase_letter}", dark_mode)
            
            # Create preview plot
            preview_plot = create_preview_plot(data, dark_mode)
            
            status_msg = create_status_message(
                f"Loaded {len(data['current']):,} samples from database", "success", dark_mode
            )
            
            return json_data, summary, preview_plot, status_msg, {'opacity': '1'}
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_msg = create_status_message(error_msg, "danger", dark_mode)
            return None, None, None, status_msg, {'opacity': '1'}
    
    elif trigger == 'upload-data':
        # Load from file
        if file_contents is None:
            status_msg = create_status_message("No file selected", "warning", dark_mode)
            return None, None, None, status_msg, {'opacity': '1'}
        
        try:
            # Parse file based on format
            data = FileHandler.parse_uploaded_file(file_contents, filename, file_format)
            
            if data is None:
                status_msg = create_status_message(
                    "Could not parse file. Ensure it contains time and current columns or has the correct format.", 
                    "warning",
                    dark_mode
                )
                return None, None, None, status_msg, {'opacity': '1'}
            
            # Convert data to JSON serializable format
            json_data = convert_to_json_serializable(data)
            
            # Create data summary
            summary = create_data_summary(data, f"File: {filename}", dark_mode)
            
            # Create preview plot
            preview_plot = create_preview_plot(data, dark_mode)
            
            status_msg = create_status_message(
                f"Loaded {len(data['current']):,} samples from file", "success", dark_mode
            )
            
            return json_data, summary, preview_plot, status_msg, {'opacity': '1'}
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            status_msg = create_status_message(error_msg, "danger", dark_mode)
            return None, None, None, status_msg, {'opacity': '1'}
    
    empty_status_style = {
        'padding': '10px 20px',
        'borderRadius': '5px',
        'marginTop': '20px',
        'marginBottom': '20px',
        'opacity': '0'  # Hidden
    }
    return None, None, None, "", empty_status_style


# Callback to update visualization based on selected type
@app.callback(
    Output('visualization-area', 'children'),
    Output('visualization-store', 'data'),
    Input('update-viz-button', 'n_clicks'),
    [State('viz-type', 'value'),
     State('current-data', 'data'),
     State('harmonics-data', 'data'),
     State('wavelet-data', 'data'),
     State('power-quality-data', 'data'),
     State('transient-data', 'data'),
     State('multi-phase-data', 'data'),
     State('theme-store', 'data')],
    prevent_initial_call=True
)
def update_visualization(n_clicks, viz_type, data_json, harmonics_data, wavelet_data, 
                         power_quality_data, transient_data, multi_phase_data, theme_data):
    if n_clicks is None or data_json is None:
        raise dash.exceptions.PreventUpdate
        
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    data = convert_from_json(data_json)
    
    if viz_type == 'time':
        fig = plot_generator.create_time_domain_plot(data, dark_mode=dark_mode)
        return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'time'}
        
    elif viz_type == 'fft':
        fft_data = signal_processor.compute_fft(data)
        fig = plot_generator.create_frequency_domain_plot(fft_data, dark_mode=dark_mode)
        return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'fft'}
        
    elif viz_type == 'harmonics':
        if harmonics_data:
            h_data = convert_from_json(harmonics_data)
            fig = plot_generator.create_harmonics_plot(h_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'harmonics'}
        else:
            h_data = signal_processor.compute_harmonics(data)
            fig = plot_generator.create_harmonics_plot(h_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'harmonics'}
            
    elif viz_type == 'wavelet':
        if wavelet_data:
            w_data = convert_from_json(wavelet_data)
            fig = plot_generator.create_wavelet_plot(w_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'wavelet'}
        else:
            w_data = signal_processor.compute_wavelet(data)
            fig = plot_generator.create_wavelet_plot(w_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'wavelet'}
            
    elif viz_type == 'stft':
        stft_data = signal_processor.compute_stft(data)
        fig = plot_generator.create_stft_plot(stft_data, dark_mode=dark_mode)
        return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'stft'}
        
    elif viz_type == 'spectrogram':
        spec_data = signal_processor.compute_spectrogram(data)
        fig = plot_generator.create_spectrogram_plot(spec_data, dark_mode=dark_mode)
        return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'spectrogram'}
        
    elif viz_type == 'transients':
        if transient_data:
            t_data = convert_from_json(transient_data)
            fig = plot_generator.create_transient_plot(t_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'transients'}
        else:
            t_data = signal_processor.analyze_transients(data)
            fig = plot_generator.create_transient_plot(t_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'transients'}
    
    elif viz_type == '3d_harmonics':
        if data_json:
            data = convert_from_json(data_json)
            fig = plot_generator.create_3d_harmonic_visualization(data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': '3d_harmonics'}
        else:
            return html.Div("Please load data first to generate 3D visualization"), None

    elif viz_type == 'multi_phase':
        if multi_phase_data:
            mp_data = convert_from_json(multi_phase_data)
            # Determine appropriate plot based on the analysis type
            if mp_data['type'] == 'thd':
                fig = plot_generator.create_multi_phase_harmonic_plot(mp_data, dark_mode=dark_mode)
            elif mp_data['type'] == 'fft':
                fig = plot_generator.create_multi_phase_fft_plot(mp_data, dark_mode=dark_mode)
            elif mp_data['type'] == 'time':
                fig = plot_generator.create_multi_phase_time_plot(mp_data, dark_mode=dark_mode)
            elif mp_data['type'] == 'correlation':
                fig = plot_generator.create_multi_phase_correlation_plot(mp_data, dark_mode=dark_mode)
            return dcc.Graph(figure=fig, config={'displayModeBar': True}), {'type': 'multi_phase'}
        else:
            # For visualization only, create a simple demo
            return html.Div("Multi-phase visualization requires running the analysis first"), None
            
    return html.Div("Select a visualization type and click Update"), None

# Callback for exporting visualizations
@app.callback(
    Output('download-data', 'data'),
    Input('export-button', 'n_clicks'),
    [State('export-format', 'value'),
     State('visualization-store', 'data'),
     State('current-data', 'data')],
    prevent_initial_call=True
)
def export_visualization(n_clicks, export_format, viz_data, current_data):
    if n_clicks is None or viz_data is None:
        raise dash.exceptions.PreventUpdate
        
    if export_format == 'csv' and current_data:
        data = convert_from_json(current_data)
        df = pd.DataFrame({
            'time': data['time'],
            'current': data['current']
        })
        return dcc.send_data_frame(df.to_csv, "waveform_data.csv", index=False)
        
    # For image exports, we need to use clientside callbacks (not shown here)
    # This would typically be implemented with a clientside JavaScript callback
    
    return dash.no_update

# Add this callback to app.py
@app.callback(
    Output('report-preview', 'children'),
    Input('generate-report-button', 'n_clicks'),
    [State('report-title', 'value'),
     State('report-include', 'value'),
     State('current-data', 'data'),
     State('analysis-results-store', 'data'),
     State('harmonics-data', 'data'),
     State('wavelet-data', 'data'),
     State('power-quality-data', 'data'),
     State('transient-data', 'data'),
     State('multi-phase-data', 'data'),
     State('theme-store', 'data')],
    prevent_initial_call=True
)
def generate_report(n_clicks, report_title, included_sections, data_json, 
                   analysis_results, harmonics_data, wavelet_data, 
                   power_quality_data, transient_data, multi_phase_data, theme_data):
    if not n_clicks or not data_json or not included_sections:
        return "Select analyses and click Generate Report to preview."
    
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    
    # Convert data from JSON
    data = convert_from_json(data_json)
    
    # Create report elements
    report_elements = []
    
    # Add title
    if not report_title:
        report_title = "WaveformPro Analysis Report"
    
    report_elements.append(html.H2(report_title, style={'marginBottom': '20px'}))
    
    # Add date
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_elements.append(html.P(f"Generated: {now}", style={'marginBottom': '30px'}))
    
    # Add sections based on user selection
    if 'time' in included_sections:
        report_elements.append(html.H3("Time Domain Analysis", style={'marginTop': '30px'}))
        metrics = signal_processor.calculate_metrics(data)
        report_elements.append(create_metrics_table(metrics, dark_mode))
        fig = plot_generator.create_time_domain_plot(data, dark_mode=dark_mode)
        report_elements.append(dcc.Graph(figure=fig))
    
    if 'fft' in included_sections:
        report_elements.append(html.H3("Frequency Domain Analysis", style={'marginTop': '30px'}))
        fft_data = signal_processor.compute_fft(data)
        fig = plot_generator.create_frequency_domain_plot(fft_data, dark_mode=dark_mode)
        report_elements.append(dcc.Graph(figure=fig))
    
    if 'harmonics' in included_sections and harmonics_data:
        report_elements.append(html.H3("Harmonics Analysis", style={'marginTop': '30px'}))
        h_data = convert_from_json(harmonics_data)
        report_elements.append(html.P(f"Total Harmonic Distortion (THD): {h_data['thd']:.2f}%"))
        fig = plot_generator.create_harmonics_plot(h_data, dark_mode=dark_mode)
        report_elements.append(dcc.Graph(figure=fig))
    
    # Add more sections for other analysis types
    # ...
    
    return report_elements

# Callback to update analysis parameters based on analysis type
@app.callback(
    Output('analysis-parameters', 'children'),
    Input('analysis-type', 'value'),
    State('theme-store', 'data')
)
def update_analysis_parameters(analysis_type, theme_data):
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    
    dropdown_style = {
        'backgroundColor': DARK_MODE_STYLES['input_bg'] if dark_mode else LIGHT_MODE_STYLES['input_bg'],
        'color': DARK_MODE_STYLES['input_text'] if dark_mode else LIGHT_MODE_STYLES['input_text'],
        'border': f'1px solid {DARK_MODE_STYLES["tab_inactive"] if dark_mode else LIGHT_MODE_STYLES["tab_inactive"]}',
        'borderRadius': '5px'
    }
    
    if analysis_type == 'time':
        return html.Div([
            html.H5("Time Domain Parameters", style={'marginBottom': '15px'}),
            html.Label("Options:"),
            dbc.Checklist(
                id='time-options-ui',
                options=[
                    {'label': 'Show Mean Line', 'value': 'show_mean'},
                    {'label': 'Show RMS Line', 'value': 'show_rms'},
                    {'label': 'Show Envelope', 'value': 'show_envelope'},
                ],
                value=['show_mean', 'show_rms'],
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            ),
        ])
    
    elif analysis_type == 'fft':
        return html.Div([
            html.H5("FFT Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Window Type:", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='fft-window-type-ui',
                options=[
                    {'label': 'Rectangular', 'value': 'rectangular'},
                    {'label': 'Hanning', 'value': 'hann'},
                    {'label': 'Hamming', 'value': 'hamming'},
                    {'label': 'Blackman', 'value': 'blackman'},
                    {'label': 'Flattop', 'value': 'flattop'}
                ],
                value='hann',
                style=dropdown_style,
                className='mb-3'
            ),
            
            html.Label("Scale:", style={'marginBottom': '5px'}),
            dbc.RadioItems(
                id='fft-scale-ui',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'dB', 'value': 'db'},
                    {'label': 'Log-Log', 'value': 'log-log'}
                ],
                value='db',
                inline=True,
                className='mb-3'
            ),
            
            html.Label("Max Frequency (Hz):", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='fft-max-freq-ui',
                min=50,
                max=2000,
                step=50,
                value=500,
                marks={i: str(i) for i in range(0, 2001, 500)},
                className='mb-3'
            )
        ])
    
    elif analysis_type == 'harmonics':
        return html.Div([
            html.H5("Harmonics Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Fundamental Frequency (Hz):", style={'marginBottom': '5px'}),
            dbc.Input(
                id='fundamental-freq-ui',
                type='number',
                value=60,
                min=1,
                max=1000,
                step=0.1,
                style={**dropdown_style, 'width': '100%'},
                className='mb-3'
            ),
            
            html.Label("Harmonics to Show:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='harmonics-count-ui',
                min=5,
                max=40,
                step=5,
                value=15,
                marks={i: str(i) for i in [5, 10, 15, 20, 30, 40]},
                className='mb-3'
            ),
            
            html.Label("View:", style={'marginBottom': '5px'}),
            dbc.RadioItems(
                id='harmonics-view-ui',
                options=[
                    {'label': 'Harmonic Spectrum', 'value': 'spectrum'},
                    {'label': 'Reconstruction', 'value': 'reconstruction'},
                    {'label': 'Components', 'value': 'components'}
                ],
                value='spectrum',
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            ),
            
            html.Div([
                html.Label("Harmonics for Reconstruction:", style={'marginBottom': '5px', 'marginTop': '15px'}),
                dbc.Checklist(
                    id='harmonics-selection-ui',
                    options=[
                        {'label': f'H{i}', 'value': i} for i in range(1, 11)
                    ],
                    value=[1, 2, 3, 4, 5, 6, 7],
                    inline=True
                )
            ], id='harmonics-selection-container', style={'display': 'none'})
        ])
    
    elif analysis_type == 'power_quality':
        return html.Div([
            html.H5("Power Quality Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Analysis Type:", style={'marginBottom': '5px'}),
            dbc.RadioItems(
                id='power-quality-type-ui',
                options=[
                    {'label': 'Flicker', 'value': 'flicker'},
                ],
                value='flicker',
                labelStyle={'display': 'block', 'marginBottom': '8px'},
                className='mb-3'
            ),
            
            html.Label("Sensitivity:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='power-quality-sensitivity-ui',
                min=1,
                max=5,
                step=1,
                value=2,
                marks={i: str(i) for i in range(1, 6)},
                className='mb-3'
            )
        ])
    
    elif analysis_type == 'transients':
        return html.Div([
            html.H5("Transient Analysis Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Sensitivity:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='transient-sensitivity-ui',
                min=1,
                max=5,
                step=1,
                value=2,
                marks={i: str(i) for i in range(1, 6)},
                className='mb-3'
            ),
            
            html.Label("Window Size (samples):", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='transient-window-ui',
                min=5,
                max=50,
                step=5,
                value=20,
                marks={i: str(i) for i in range(5, 51, 5)},
                className='mb-3'
            )
        ])
    
    elif analysis_type == 'wavelet':
        return html.Div([
            html.H5("Wavelet Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Wavelet Type:", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='wavelet-type-ui',
                options=[
                    {'label': 'Daubechies 4', 'value': 'db4'},
                    {'label': 'Daubechies 8', 'value': 'db8'},
                    {'label': 'Symlet 5', 'value': 'sym5'},
                    {'label': 'Coiflet 3', 'value': 'coif3'},
                    {'label': 'Biorthogonal 3.5', 'value': 'bior3.5'}
                ],
                value='db4',
                style=dropdown_style,
                className='mb-3'
            ),
            
            html.Label("Decomposition Level:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='wavelet-level-ui',
                min=1,
                max=8,
                step=1,
                value=2,
                marks={i: str(i) for i in range(1, 9)},
                className='mb-3'
            )
        ])
    
    elif analysis_type == 'stft':
        return html.Div([
            html.H5("STFT Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Window Size:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='stft-window-size-ui',
                min=64,
                max=512,
                step=64,
                value=256,
                marks={i: str(i) for i in range(64, 513, 64)},
                className='mb-3'
            ),
            
            html.Label("Overlap (%):", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='stft-overlap-ui',
                min=0,
                max=75,
                step=25,
                value=50,
                marks={i: f"{i}%" for i in range(0, 76, 25)},
                className='mb-3'
            ),
            
            html.Label("Window Type:", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='stft-window-type-ui',
                options=[
                    {'label': 'Hanning', 'value': 'hann'},
                    {'label': 'Hamming', 'value': 'hamming'},
                    {'label': 'Blackman', 'value': 'blackman'},
                    {'label': 'Rectangular', 'value': 'boxcar'}
                ],
                value='hann',
                style=dropdown_style,
                className='mb-3'
            ),
            
            html.Label("Colormap:", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='stft-colormap-ui',
                options=[
                    {'label': 'Viridis', 'value': 'viridis'},
                    {'label': 'Plasma', 'value': 'plasma'},
                    {'label': 'Inferno', 'value': 'inferno'},
                    {'label': 'Jet', 'value': 'jet'},
                    {'label': 'Turbo', 'value': 'turbo'}
                ],
                value='viridis',
                style=dropdown_style
            )
        ])
    
    elif analysis_type == 'cepstrum':
        return html.Div([
            html.H5("Cepstrum Analysis Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Window Type:", style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='cepstrum-window-type-ui',
                options=[
                    {'label': 'Hanning', 'value': 'hann'},
                    {'label': 'Hamming', 'value': 'hamming'},
                    {'label': 'Blackman', 'value': 'blackman'},
                    {'label': 'Rectangular', 'value': 'rectangular'}
                ],
                value='hann',
                style=dropdown_style,
                className='mb-3'
            ),
            
            html.Label("Peak Detection Threshold (%):", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='cepstrum-threshold-ui',
                min=5,
                max=50,
                step=5,
                value=10,
                marks={i: f"{i}%" for i in range(5, 51, 5)},
                className='mb-3'
            )
        ])
    
    elif analysis_type == 'distortion':
        return html.Div([
            html.H5("Waveform Distortion Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Fundamental Frequency (Hz):", style={'marginBottom': '5px'}),
            dbc.Input(
                id='distortion-fund-freq-ui',
                type='number',
                value=60,
                min=1,
                max=1000,
                step=0.1,
                style={**dropdown_style, 'width': '100%'},
                className='mb-3'
            ),
            
            html.Label("Harmonics to Calculate:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='distortion-harmonics-ui',
                min=10,
                max=50,
                step=10,
                value=40,
                marks={i: str(i) for i in range(10, 51, 10)},
                className='mb-3'
            ),
            
            html.Label("Display Options:", style={'marginBottom': '5px'}),
            dbc.Checklist(
                id='distortion-options-ui',
                options=[
                    {'label': 'Show Even Harmonics', 'value': 'show_even'},
                    {'label': 'Show Odd Harmonics', 'value': 'show_odd'},
                    {'label': 'Show Triplen Harmonics', 'value': 'show_triplen'},
                    {'label': 'Show K-Factor Details', 'value': 'show_k_factor'}
                ],
                value=['show_even', 'show_odd', 'show_triplen'],
                labelStyle={'display': 'block', 'marginBottom': '8px'}
            )
        ])
    
    elif analysis_type == 'multi_phase':
        return html.Div([
            html.H5("Multi-Phase Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Phases to Display:", style={'marginBottom': '5px'}),
            dbc.Checklist(
                id='multi-phase-selection-ui',
                options=[
                    {'label': f'Phase A', 'value': '1'},
                    {'label': f'Phase B', 'value': '2'},
                    {'label': f'Phase C', 'value': '3'}
                ],
                value=['1', '2', '3'],
                labelStyle={'display': 'block', 'marginBottom': '8px'},
                className='mb-3'
            ),
            
            html.Label("Analysis Type:", style={'marginBottom': '5px'}),
            dbc.RadioItems(
                id='multi-phase-type-ui',
                options=[
                    {'label': 'Time Domain', 'value': 'time'},
                    {'label': 'FFT Comparison', 'value': 'fft'},
                    {'label': 'THD Comparison', 'value': 'thd'},
                    {'label': 'Phase Correlation', 'value': 'correlation'},
                    {'label': 'Impedance Analysis', 'value': 'impedance'}
                ],
                value='thd',
                labelStyle={'display': 'block', 'marginBottom': '8px'},
                className='mb-3'
            ),
            
            html.Label("Plot Style:", style={'marginBottom': '5px'}),
            dbc.RadioItems(
                id='multi-phase-plot-style-ui',
                options=[
                    {'label': 'Overlay', 'value': 'overlay'},
                    {'label': 'Separate', 'value': 'separate'}
                ],
                value='overlay',
                inline=True
            )
        ])
    
    elif analysis_type == 'coherence':
        return html.Div([
            html.H5("Coherence Analysis Parameters", style={'marginBottom': '15px'}),
            
            html.Label("Segment Length:", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='coherence-segment-length-ui',
                min=256,
                max=2048,
                step=256,
                value=1024,
                marks={i: str(i) for i in range(256, 2049, 256)},
                className='mb-3'
            ),
            
            html.Label("Overlap (%):", style={'marginBottom': '5px'}),
            dcc.Slider(
                id='coherence-overlap-ui',
                min=0,
                max=75,
                step=25,
                value=50,
                marks={i: f"{i}%" for i in range(0, 76, 25)},
                className='mb-3'
            ),
            
            html.Label("Frequency Range:", style={'marginBottom': '5px'}),
            dcc.RangeSlider(
                id='coherence-freq-range-ui',
                min=0,
                max=1000,
                step=50,
                value=[0, 500],
                marks={i: str(i) for i in range(0, 1001, 200)},
                className='mb-3'
            )
        ])
        
    elif analysis_type == 'symmetrical':
        return html.Div([
            html.H5("Symmetrical Components Parameters", style={'marginBottom': '15px'}),
            
            html.Label("This analysis requires data from all three phases."),
            html.P("The analysis will be performed on the fundamental component (60Hz) of each phase."),
            
            html.Label("Phase Mapping:", style={'marginTop': '15px', 'marginBottom': '5px'}),
            html.Div([
                html.Label("Phase A:"),
                dcc.Dropdown(
                    id='symmetrical-phaseA-ui',
                    options=[{'label': f'Phase {i}', 'value': str(i)} for i in range(1, 4)],
                    value='1',
                    style=dropdown_style,
                    className='mb-2'
                ),
                html.Label("Phase B:"),
                dcc.Dropdown(
                    id='symmetrical-phaseB-ui',
                    options=[{'label': f'Phase {i}', 'value': str(i)} for i in range(1, 4)],
                    value='2',
                    style=dropdown_style,
                    className='mb-2'
                ),
                html.Label("Phase C:"),
                dcc.Dropdown(
                    id='symmetrical-phaseC-ui',
                    options=[{'label': f'Phase {i}', 'value': str(i)} for i in range(1, 4)],
                    value='3',
                    style=dropdown_style
                )
            ])
        ])
    
    elif analysis_type == 'pqi':
        return html.Div([
            html.H5("Power Quality Index Parameters", style={'marginBottom': '15px'}),
            
            html.P("The Power Quality Index (PQI) combines multiple metrics to assess overall power quality."),
            html.P("The following components are used to calculate the index:"),
            
            html.Ul([
                html.Li("Crest Factor (25%): Measures how peaked the waveform is"),
                html.Li("Form Factor (20%): Measures the waveform shape"),
                html.Li("THD (40%): Total Harmonic Distortion"),
                html.Li("Transients (15%): Number of transient events")
            ])
        ])
    
    return html.Div("Select an analysis type")

# Callback to update hidden stores based on UI inputs
@app.callback(
    Output('time-options', 'data'),
    Input('time-options-ui', 'value')
)
def update_time_options_store(value):
    return value if value else ['show_mean', 'show_rms']

@app.callback(
    Output('fft-window-type', 'data'),
    Input('fft-window-type-ui', 'value')
)
def update_fft_window_type_store(value):
    return value if value else 'hann'

@app.callback(
    Output('fft-scale', 'data'),
    Input('fft-scale-ui', 'value')
)
def update_fft_scale_store(value):
    return value if value else 'db'

@app.callback(
    Output('fft-max-freq', 'data'),
    Input('fft-max-freq-ui', 'value')
)
def update_fft_max_freq_store(value):
    return value if value else 500

@app.callback(
    Output('harmonics-count', 'data'),
    Input('harmonics-count-ui', 'value')
)
def update_harmonics_count_store(value):
    return value if value else 15

@app.callback(
    Output('fundamental-freq', 'data'),
    Input('fundamental-freq-ui', 'value')
)
def update_fundamental_freq_store(value):
    return value if value else 60

@app.callback(
    Output('wavelet-type', 'data'),
    Input('wavelet-type-ui', 'value')
)
def update_wavelet_type_store(value):
    return value if value else 'db4'

@app.callback(
    Output('wavelet-level', 'data'),
    Input('wavelet-level-ui', 'value')
)
def update_wavelet_level_store(value):
    return value if value else 2

@app.callback(
    Output('power-quality-type', 'data'),
    Input('power-quality-type-ui', 'value')
)
def update_power_quality_type_store(value):
    return value if value else 'flicker'

@app.callback(
    Output('power-quality-sensitivity', 'data'),
    Input('power-quality-sensitivity-ui', 'value')
)
def update_power_quality_sensitivity_store(value):
    return value if value else 2

@app.callback(
    Output('transient-sensitivity', 'data'),
    Input('transient-sensitivity-ui', 'value')
)
def update_transient_sensitivity_store(value):
    return value if value else 2

@app.callback(
    Output('transient-window', 'data'),
    Input('transient-window-ui', 'value')
)
def update_transient_window_store(value):
    return value if value else 20

@app.callback(
    Output('multi-phase-selection', 'data'),
    Input('multi-phase-selection-ui', 'value')
)
def update_multi_phase_selection_store(value):
    return value if value else ['1', '2', '3']

@app.callback(
    Output('multi-phase-type', 'data'),
    Input('multi-phase-type-ui', 'value')
)
def update_multi_phase_type_store(value):
    return value if value else 'thd'

@app.callback(
    Output('multi-phase-plot-style', 'data'),
    Input('multi-phase-plot-style-ui', 'value')
)
def update_multi_phase_plot_style_store(value):
    return value if value else 'overlay'

@app.callback(
    Output('stft-window-size', 'data'),
    Input('stft-window-size-ui', 'value')
)
def update_stft_window_size_store(value):
    return value if value else 256

@app.callback(
    Output('stft-overlap', 'data'),
    Input('stft-overlap-ui', 'value')
)
def update_stft_overlap_store(value):
    return value if value else 50

@app.callback(
    Output('stft-window-type', 'data'),
    Input('stft-window-type-ui', 'value')
)
def update_stft_window_type_store(value):
    return value if value else 'hann'

@app.callback(
    Output('stft-colormap', 'data'),
    Input('stft-colormap-ui', 'value')
)
def update_stft_colormap_store(value):
    return value if value else 'viridis'

@app.callback(
    Output('harmonics-view', 'data'),
    Input('harmonics-view-ui', 'value')
)
def update_harmonics_view_store(value):
    return value if value else 'spectrum'

@app.callback(
    Output('harmonics-selection', 'data'),
    Input('harmonics-selection-ui', 'value')
)
def update_harmonics_selection_store(value):
    return value if value else [1, 2, 3, 4, 5, 6, 7]

@app.callback(
    Output('interharmonics-fund-freq', 'data'),
    Input('interharmonics-fund-freq-ui', 'value')
)
def update_interharmonics_fund_freq_store(value):
    return value if value else 60

# Add callbacks for new analysis parameters
@app.callback(
    Output('coherence-segment-length', 'data'),
    Input('coherence-segment-length-ui', 'value')
)
def update_coherence_segment_length_store(value):
    return value if value else 1024

@app.callback(
    Output('coherence-overlap', 'data'),
    Input('coherence-overlap-ui', 'value')
)
def update_coherence_overlap_store(value):
    return value if value else 50

@app.callback(
    Output('coherence-freq-range', 'data'),
    Input('coherence-freq-range-ui', 'value')
)
def update_coherence_freq_range_store(value):
    return value if value else [0, 500]

@app.callback(
    Output('symmetrical-phaseA', 'data'),
    Input('symmetrical-phaseA-ui', 'value')
)
def update_symmetrical_phaseA_store(value):
    return value if value else '1'

@app.callback(
    Output('symmetrical-phaseB', 'data'),
    Input('symmetrical-phaseB-ui', 'value')
)
def update_symmetrical_phaseB_store(value):
    return value if value else '2'

@app.callback(
    Output('symmetrical-phaseC', 'data'),
    Input('symmetrical-phaseC-ui', 'value')
)
def update_symmetrical_phaseC_store(value):
    return value if value else '3'

# Show/hide harmonics selection based on view
@app.callback(
    Output('harmonics-selection-container', 'style'),
    Input('harmonics-view-ui', 'value')
)
def toggle_harmonics_selection(view):
    if view == 'reconstruction':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Callback to run analysis
@app.callback(
    Output('analysis-results', 'children'),
    Output('analysis-results-store', 'data'),
    Output('harmonics-data', 'data'),
    Output('wavelet-data', 'data'),
    Output('power-quality-data', 'data'),
    Output('transient-data', 'data'),
    Output('multi-phase-data', 'data'),
    Output('status-area', 'children', allow_duplicate=True),
    Output('status-area', 'style', allow_duplicate=True),
    Input('run-analysis-button', 'n_clicks'),
    [State('current-data', 'data'),
     State('analysis-type', 'value'),
     State('time-options', 'data'),
     State('fft-window-type', 'data'),
     State('fft-scale', 'data'),
     State('fft-max-freq', 'data'),
     State('harmonics-count', 'data'),
     State('fundamental-freq', 'data'),
     State('harmonics-view', 'data'),
     State('harmonics-selection', 'data'),
     State('wavelet-type', 'data'),
     State('wavelet-level', 'data'),
     State('power-quality-type', 'data'),
     State('power-quality-sensitivity', 'data'),
     State('interharmonics-fund-freq', 'data'),
     State('transient-sensitivity', 'data'),
     State('transient-window', 'data'),
     State('stft-window-size', 'data'),
     State('stft-overlap', 'data'),
     State('stft-window-type', 'data'),
     State('stft-colormap', 'data'),
     State('multi-phase-selection', 'data'),
     State('multi-phase-type', 'data'),
     State('multi-phase-plot-style', 'data'),
     State('coherence-segment-length', 'data'),
     State('coherence-overlap', 'data'),
     State('coherence-freq-range', 'data'),
     State('symmetrical-phaseA', 'data'),
     State('symmetrical-phaseB', 'data'),
     State('symmetrical-phaseC', 'data'),
     State('theme-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, data_json, analysis_type='time', 
                time_options=None, fft_window_type='hann', fft_scale='db', fft_max_freq=500,
                harmonics_count=15, fundamental_freq=60, harmonics_view='spectrum', harmonics_selection=None,
                wavelet_type='db4', wavelet_level=2,
                power_quality_type='flicker', power_quality_sensitivity=2,
                interharmonics_fund_freq=60,
                transient_sensitivity=2, transient_window=20,
                stft_window_size=256, stft_overlap=50, stft_window_type='hann', stft_colormap='viridis',
                multi_phase_selection=None, multi_phase_type='thd', multi_phase_plot_style='overlay',
                coherence_segment_length=1024, coherence_overlap=50, coherence_freq_range=None,
                symmetrical_phaseA='1', symmetrical_phaseB='2', symmetrical_phaseC='3',
                theme_data=None):
    
    # Initialize defaults
    if time_options is None:
        time_options = ['show_mean', 'show_rms']
    if harmonics_selection is None:
        harmonics_selection = [1, 2, 3, 4, 5, 6, 7]
    if multi_phase_selection is None:
        multi_phase_selection = ['1', '2', '3']
    if coherence_freq_range is None:
        coherence_freq_range = [0, 500]

    if n_clicks is None or data_json is None:
        raise dash.exceptions.PreventUpdate
    
    # Get dark mode state
    dark_mode = theme_data.get('dark_mode', False) if theme_data else False
    
    # Convert data from JSON
    data = convert_from_json(data_json)
    if data is None:
        return html.Div("No data available for analysis"), None, None, None, None, None, None, create_status_message(
            "No data available for analysis", "warning", dark_mode
        ), {'opacity': '1'}
    
    try:
        # Initialize results
        harmonics_data_result = None
        wavelet_data_result = None
        power_quality_data_result = None
        transient_data_result = None
        multi_phase_data_result = None
        
        if analysis_type == 'time':
            # Time domain analysis
            metrics = signal_processor.calculate_metrics(data)
            
            # Create time domain plot
            fig = plot_generator.create_time_domain_plot(
                data, 
                show_mean='show_mean' in time_options if time_options else True,
                show_rms='show_rms' in time_options if time_options else True,
                show_envelope='show_envelope' in time_options if time_options else False,
                dark_mode=dark_mode
            )
            
            # Create metrics table
            metrics_table = create_metrics_table(metrics, dark_mode)
            
            # Combine in a card
            results_html = html.Div([
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True, 'scrollZoom': True}
                    )
                ], style={'marginBottom': '20px'}),
                
                metrics_table
            ])
            
            # Store results
            results_data = {'type': 'time', 'metrics': convert_to_json_serializable(metrics)}
            
            status_msg = create_status_message("Time domain analysis completed", "success", dark_mode)
            
        elif analysis_type == 'fft':
            # FFT analysis
            fft_data = signal_processor.compute_fft(data, window_type=fft_window_type)
            
            # Create FFT plot
            fig = plot_generator.create_frequency_domain_plot(
                fft_data, 
                scale=fft_scale,
                max_freq=fft_max_freq,
                dark_mode=dark_mode
            )
            
            # Combine in a card
            results_html = html.Div([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True, 'scrollZoom': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'fft', 'fft_data': convert_to_json_serializable(fft_data)}
            
            status_msg = create_status_message("FFT analysis completed", "success", dark_mode)
            
        elif analysis_type == 'harmonics':
            # Harmonics analysis
            harmonics_data = signal_processor.compute_harmonics(
                data, 
                fundamental_freq=fundamental_freq,
                num_harmonics=40  # Always calculate all, then display based on user selection
            )
            
            # Store harmonics data for reuse
            harmonics_data_result = convert_to_json_serializable(harmonics_data)
            
            if harmonics_view == 'spectrum':
                # Create harmonics spectrum plot
                fig = plot_generator.create_harmonics_plot(
                    harmonics_data,
                    num_harmonics=harmonics_count,
                    dark_mode=dark_mode
                )
                
                results_html = html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
            
            elif harmonics_view == 'reconstruction':
                # Create harmonics reconstruction plot
                fig = plot_generator.create_signal_reconstruction_plot(
                    harmonics_data,
                    selected_harmonics=harmonics_selection,
                    dark_mode=dark_mode
                )
                
                # Create harmonics selection controls
                harmonics_controls = html.Div([
                    html.H5("Select Harmonics for Reconstruction", style={'marginBottom': '15px'}),
                    dbc.Checklist(
                        id='harmonics-selection-ui',
                        options=[{'label': f'H{i}', 'value': i} for i in range(1, 16)],
                        value=harmonics_selection,
                        inline=True
                    )
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                results_html = html.Div([
                    harmonics_controls,
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
            
            elif harmonics_view == 'components':
                # Create harmonics components plot
                fig = plot_generator.create_harmonic_components_plot(
                    harmonics_data,
                    num_harmonics=min(harmonics_count, 10),  # Limit to 10 components max
                    dark_mode=dark_mode
                )
                
                results_html = html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
            
            else:
                # Default to spectrum view
                fig = plot_generator.create_harmonics_plot(
                    harmonics_data,
                    num_harmonics=harmonics_count,
                    dark_mode=dark_mode
                )
                
                results_html = html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
            
            # Store results
            results_data = {'type': 'harmonics', 'view': harmonics_view}
            
            status_msg = create_status_message("Harmonics analysis completed", "success", dark_mode)
            
        elif analysis_type == 'wavelet':
            # Wavelet analysis
            wavelet_data = signal_processor.compute_wavelet(
                data,
                wavelet=wavelet_type,
                level=wavelet_level
            )
            
            # Store wavelet data for reuse
            wavelet_data_result = convert_to_json_serializable(wavelet_data)
            
            # Create wavelet plot
            fig = plot_generator.create_wavelet_plot(
                wavelet_data,
                dark_mode=dark_mode
            )
            
            results_html = html.Div([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'wavelet'}
            
            status_msg = create_status_message("Wavelet analysis completed", "success", dark_mode)
            
        elif analysis_type == 'coherence':
            # Get parameters
            segment_length = coherence_segment_length if coherence_segment_length else 1024
            overlap = int(segment_length * (coherence_overlap / 100)) if coherence_overlap else segment_length // 2
            
            # Compute coherence
            coherence_data = signal_processor.compute_coherence(
                data,
                segment_length=segment_length,
                overlap=overlap
            )
            
            if coherence_data is None:
                return html.Div("Error computing coherence analysis"), None, None, None, None, None, None, create_status_message("Error in coherence analysis", "danger", dark_mode), {'opacity': '1'}
            
            # Create coherence plot
            fig = plot_generator.create_coherence_plot(
                coherence_data,
                dark_mode=dark_mode
            )
            
            # Create summary card
            bands_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Frequency Band"),
                    html.Th("Amp Coherence"),
                    html.Th("Phase Coherence")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(band['band']),
                        html.Td(f"{band['amplitude_coherence']['mean']:.3f}"),
                        html.Td(f"{band['phase_coherence']['mean']:.3f}")
                    ]) for band in coherence_data['band_analysis']
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'marginBottom': '20px'
            })
            
            summary_card = html.Div([
                html.H5("Coherence Analysis Summary", style={'marginBottom': '15px'}),
                html.P("Coherence measures the linear relationship between different components of the signal."),
                bands_table
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
            
            results_html = html.Div([
                summary_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'coherence'}
            
            status_msg = create_status_message("Coherence analysis completed", "success", dark_mode)
            
        elif analysis_type == 'symmetrical':
            # For now, we use a similar approach to multi-phase analysis to simulate having data from all 3 phases
            # In a real application, you would load data for all phases
            
            # Create simulated phase data
            phase_data = {}
            for phase in ['1', '2', '3']:
                # Create slightly different data for each phase for demonstration
                phase_copy = dict(data)
                
                # Add phase shift and amplitude variation
                phase_int = int(phase)
                phase_shift = (phase_int - 1) * np.pi / 3  # 60 degrees between phases
                amplitude_factor = 0.9 + phase_int * 0.1  # Slight amplitude differences
                
                # Create balanced signals as the base
                phase_copy['current'] = amplitude_factor * data['current'] * np.cos(2 * np.pi * 60 * data['time'] + phase_shift)
                
                # Add some unbalance to Phase 3 to make the analysis interesting
                if phase == '3':
                    # Reduce amplitude by 5%
                    phase_copy['current'] *= 0.95
                    # Add slight phase shift
                    phase_copy['current'] = phase_copy['current'] * np.cos(0.1)  # Small additional phase shift
                
                phase_data[phase] = phase_copy
            
            # Compute symmetrical components
            sym_comp_data = signal_processor.compute_symmetrical_components(phase_data)
            
            if sym_comp_data is None:
                return html.Div("Error computing symmetrical components. This analysis requires data from all three phases."), None, None, None, None, None, None, create_status_message("Error in symmetrical components analysis", "danger", dark_mode), {'opacity': '1'}
            
            # Create visualization
            fig = plot_generator.create_symmetrical_components_plot(sym_comp_data, dark_mode=dark_mode)
            
            # Create summary card
            summary_card = html.Div([
                html.H5("Symmetrical Components Analysis", style={'marginBottom': '15px'}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Component"), 
                            html.Th("Magnitude"), 
                            html.Th("Angle (°)")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("Positive Sequence"),
                                html.Td(f"{sym_comp_data['positive_sequence']['magnitude']:.4f}"),
                                html.Td(f"{sym_comp_data['positive_sequence']['angle']:.2f}")
                            ]),
                            html.Tr([
                                html.Td("Negative Sequence"),
                                html.Td(f"{sym_comp_data['negative_sequence']['magnitude']:.4f}"),
                                html.Td(f"{sym_comp_data['negative_sequence']['angle']:.2f}")
                            ]),
                            html.Tr([
                                html.Td("Zero Sequence"),
                                html.Td(f"{sym_comp_data['zero_sequence']['magnitude']:.4f}"),
                                html.Td(f"{sym_comp_data['zero_sequence']['angle']:.2f}")
                            ])
                        ])
                    ], style={'width': '100%', 'border-collapse': 'collapse'})
                ]),
                html.Div([
                    html.H6("Unbalance Factors:", style={'marginTop': '15px', 'marginBottom': '5px'}),
                    html.Div(f"Negative Sequence Unbalance: {sym_comp_data['unbalance_factors']['negative_sequence']:.2f}%"),
                    html.Div(f"Zero Sequence Unbalance: {sym_comp_data['unbalance_factors']['zero_sequence']:.2f}%")
                ], style={'marginTop': '15px'})
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
            
            results_html = html.Div([
                summary_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'symmetrical'}
            
            status_msg = create_status_message("Symmetrical components analysis completed", "success", dark_mode)
            
        elif analysis_type == 'pqi':
            # First get harmonics data (for THD)
            harmonics_data = signal_processor.compute_harmonics(data)
            
            # Get transient data
            transient_data = signal_processor.analyze_transients(data)
            
            # Calculate Power Quality Index
            pqi_data = signal_processor.calculate_power_quality_index(
                data,
                harmonics_data=harmonics_data,
                transient_data=transient_data
            )
            
            if pqi_data is None:
                return html.Div("Error calculating Power Quality Index"), None, None, None, None, None, None, create_status_message("Error in PQI analysis", "danger", dark_mode), {'opacity': '1'}
            
            # Create visualization
            fig = plot_generator.create_power_quality_index_plot(pqi_data, dark_mode=dark_mode)
            
            # Create summary card
            summary_card = html.Div([
                html.H5("Power Quality Index", style={'marginBottom': '15px'}),
                html.Div([
                    html.Div(f"Overall PQI: {pqi_data['pqi']:.1f}/100", style={'fontSize': '24px', 'marginBottom': '10px'}),
                    html.Div(f"Quality Assessment: {pqi_data['quality_level']}", style={'fontSize': '18px', 'marginBottom': '15px'})
                ]),
                html.Div([
                    html.H6("Component Scores:", style={'marginTop': '15px', 'marginBottom': '5px'}),
                    html.Div([
                        html.Div(f"Crest Factor: {pqi_data['components']['crest_factor_index']:.1f}/100", style={'marginBottom': '5px'}),
                        html.Div(f"Form Factor: {pqi_data['components']['form_factor_index']:.1f}/100", style={'marginBottom': '5px'}),
                        html.Div(f"THD: {pqi_data['components']['thd_index']:.1f}/100", style={'marginBottom': '5px'}),
                        html.Div(f"Transients: {pqi_data['components']['transient_index']:.1f}/100", style={'marginBottom': '5px'})
                    ])
                ])
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
            
            results_html = html.Div([
                summary_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'pqi'}
            
            status_msg = create_status_message("Power Quality Index analysis completed", "success", dark_mode)
            
        elif analysis_type == 'interharmonics':
            # Interharmonics analysis
            # Safely get parameters from State variables
            fundamental_freq = fundamental_freq if 'fundamental_freq' in locals() else 60
            num_groups = harmonics_count if 'harmonics_count' in locals() else 10
    
            # Call the interharmonics analysis function
            interharmonics_data = signal_processor.compute_interharmonics(
                data,
                fundamental_freq=fundamental_freq,
                num_groups=num_groups
            )
    
            # Create interharmonics plot
            fig = plot_generator.create_interharmonics_plot(
                interharmonics_data,
                dark_mode=dark_mode
            )
    
            # Create interharmonics metrics
            tid = interharmonics_data['tid']
            fundamental_magnitude = interharmonics_data['fundamental_magnitude']
    
            # Create metrics card
            metrics_card = html.Div([
                html.H5("Interharmonics Analysis", style={'marginBottom': '15px'}),
                html.Div([
                    html.Div(f"Total Interharmonic Distortion (TID): {tid:.2f}%", 
                             style={'marginBottom': '8px'}),
                    html.Div(f"Fundamental Frequency: {fundamental_freq:.2f} Hz", 
                             style={'marginBottom': '8px'}),
                    html.Div(f"Fundamental Magnitude: {fundamental_magnitude:.5f}", 
                             style={'marginBottom': '8px'})
                ])
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
    
            results_html = html.Div([
                metrics_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
    
            # Store results
            results_data = {'type': 'interharmonics'}
    
            status_msg = create_status_message("Interharmonics analysis completed", "success", dark_mode)    

        elif analysis_type == 'cepstrum':
            # Cepstrum analysis
            # Safely get parameters from existing variables
            window_type = fft_window_type if 'fft_window_type' in locals() else 'hann'
            threshold = 0.1  # Default threshold value
    
            # Call the cepstrum analysis function
            cepstrum_data = signal_processor.compute_cepstrum(data)
    
            # Create cepstrum plot
            fig = plot_generator.create_cepstrum_plot(
                cepstrum_data,
                dark_mode=dark_mode
            )
    
            # Create metrics card with peak data
            peak_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Peak #", style={'textAlign': 'center'}),
                    html.Th("Quefrency (s)", style={'textAlign': 'center'}),
                    html.Th("Frequency (Hz)", style={'textAlign': 'center'}),
                    html.Th("Amplitude", style={'textAlign': 'center'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(i+1, style={'textAlign': 'center'}),
                        html.Td(f"{peak['quefrency']:.6f}", style={'textAlign': 'center'}),
                        html.Td(f"{peak['frequency']:.2f}", style={'textAlign': 'center'}),
                        html.Td(f"{peak['amplitude']:.4f}", style={'textAlign': 'center'})
                    ]) for i, peak in enumerate(cepstrum_data['peaks'][:5])  # Show top 5 peaks
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'marginBottom': '20px'
            })
    
            metrics_card = html.Div([
                html.H5("Cepstrum Analysis - Top Peaks", style={'marginBottom': '15px'}),
                peak_table,
                html.Div([
                    html.P("The cepstrum analysis detects repetitive patterns in the frequency domain. "
                          "Peaks in the cepstrum correspond to periodicities in the signal.")
                ])
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
    
            results_html = html.Div([
                metrics_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
    
            # Store results
            results_data = {'type': 'cepstrum'}
    
            status_msg = create_status_message("Cepstrum analysis completed", "success", dark_mode)

        elif analysis_type == 'distortion':
            # Waveform distortion analysis
            # Use fundamental_freq which is already available in the callback parameters
            fund_freq = fundamental_freq if 'fundamental_freq' in locals() else 60
    
            # Compute waveform distortion metrics
            distortion_data = signal_processor.compute_waveform_distortion(
                data,
                fundamental_freq=fund_freq
            )
    
            # Create distortion plot
            fig = plot_generator.create_waveform_distortion_plot(
                distortion_data,
                dark_mode=dark_mode
            )
    
            # Create metrics card with distortion data
            metrics_card = html.Div([
                html.H5("Waveform Distortion Analysis", style={'marginBottom': '15px'}),
                html.Div([
                    html.Div(f"Total Harmonic Distortion (THD): {distortion_data['thd']:.2f}%", 
                            style={'marginBottom': '8px'}),
                    html.Div(f"K-Factor: {distortion_data['k_factor']:.2f}", 
                            style={'marginBottom': '8px'}),
                    html.Div(f"Crest Factor: {distortion_data['crest_factor']:.2f}", 
                            style={'marginBottom': '8px'}),
                    html.Div(f"Form Factor: {distortion_data['form_factor']:.2f}", 
                            style={'marginBottom': '8px'}),
                    html.Div(f"Transformer Derating Factor: {distortion_data['transformer_derating_factor']:.2f}", 
                            style={'marginBottom': '8px'})
                ])
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
    
            results_html = html.Div([
                metrics_card,
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
    
            # Store results
            results_data = {'type': 'distortion'}
    
            status_msg = create_status_message("Waveform distortion analysis completed", "success", dark_mode)

        elif analysis_type == 'power_quality':
            # Power quality analysis
            power_quality_data = signal_processor.analyze_power_quality(
                data,
                analysis_type=power_quality_type,
                sensitivity=power_quality_sensitivity
            )
            
            # Store power quality data for reuse
            power_quality_data_result = convert_to_json_serializable(power_quality_data)
            
            if power_quality_type == 'flicker':
                # Create flicker plot
                fig = plot_generator.create_flicker_plot(
                    power_quality_data,
                    dark_mode=dark_mode
                )
                
                # Create flicker metrics table
                flicker_metrics = {
                    'Short-term Flicker (Pst)': power_quality_data['Pst'],
                    'Long-term Flicker (Plt)': power_quality_data['Plt'],
                    'Maximum Deviation': power_quality_data['max_deviation'],
                    'Average Deviation': power_quality_data['avg_deviation']
                }
                
                # Create metrics card
                metrics_card = html.Div([
                    html.Div("Flicker Interpretation:", style={
                        'backgroundColor': '#cfe8ff' if not dark_mode else '#17385b',
                        'color': '#0a58ca' if not dark_mode else '#ffffff',
                        'padding': '15px',
                        'borderRadius': '5px',
                        'marginBottom': '15px'
                    }),
                    html.Ul([
                        html.Li(f"Pst > 1.0 indicates visible flicker", style={'marginBottom': '8px'}),
                        html.Li(f"Plt > 0.8 suggests long-term flicker problems")
                    ], style={'paddingLeft': '20px'})
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                results_html = html.Div([
                    metrics_card,
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
            
            # Store results
            results_data = {'type': 'power_quality', 'analysis_type': power_quality_type}
            
            status_msg = create_status_message(f"{power_quality_type.capitalize()} analysis completed", "success", dark_mode)
        
        elif analysis_type == 'transients':
            # Transient analysis
            transient_data = signal_processor.analyze_transients(
                data,
                sensitivity=transient_sensitivity,
                window_size=transient_window
            )
            
            # Store transient data for reuse
            transient_data_result = convert_to_json_serializable(transient_data)
            
            # Create transient plot
            fig = plot_generator.create_transient_plot(
                transient_data,
                dark_mode=dark_mode
            )
            
            # Create events table
            events = transient_data['events']
            
            events_table = html.Table([
                html.Thead(html.Tr([
                    html.Th("#", style={'textAlign': 'center'}),
                    html.Th("Start Time (s)", style={'textAlign': 'center'}),
                    html.Th("Duration (ms)", style={'textAlign': 'center'}),
                    html.Th("Peak Value (A)", style={'textAlign': 'center'}),
                    html.Th("Classification", style={'textAlign': 'center'})
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(i+1, style={'textAlign': 'center'}),
                        html.Td(f"{event['start_time']:.5f}", style={'textAlign': 'center'}),
                        html.Td(f"{event['duration']:.2f}", style={'textAlign': 'center'}),
                        html.Td(f"{event['peak_value']:.4f}", style={'textAlign': 'center'}),
                        html.Td(f"{event['class']}", style={'textAlign': 'center'})
                    ]) for i, event in enumerate(events[:10])  # Show first 10 events
                ])
            ], style={
                'width': '100%',
                'borderCollapse': 'collapse',
                'marginBottom': '20px'
            })
            
            # Create summary metrics
            metrics_card = html.Div([
                html.H5("Transient Analysis Summary", style={'marginBottom': '15px'}),
                html.Div([
                    html.Div(f"Total Events Detected: {transient_data['detected_count']}", 
                             style={'marginBottom': '8px'}),
                    html.Div(f"Maximum Peak: {transient_data['max_peak']:.4f} A", 
                             style={'marginBottom': '8px'}),
                    html.Div(f"Average Duration: {transient_data['avg_duration']:.2f} ms", 
                             style={'marginBottom': '8px'}),
                    html.Div(f"Detection Threshold: {transient_data['threshold']:.4f} A", 
                             style={'marginBottom': '8px'})
                ])
            ], style={
                'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                'borderRadius': '5px',
                'padding': '15px',
                'marginBottom': '20px'
            })
            
            results_html = html.Div([
                metrics_card,
                html.Div([
                    html.H5("Detected Events", style={'marginBottom': '10px'}),
                    events_table if events else html.Div("No transient events detected.")
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                }),
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'transients'}
            
            status_msg = create_status_message("Transient analysis completed", "success", dark_mode)
            
        elif analysis_type == 'stft':
            # Short-time Fourier Transform analysis
            stft_data = signal_processor.compute_stft(
                data,
                nperseg=stft_window_size,
                noverlap=int(stft_window_size * stft_overlap / 100),  # Convert percentage to samples
                window=stft_window_type
            )
            
            # Create STFT plot
            fig = plot_generator.create_stft_plot(
                stft_data,
                colormap=stft_colormap,
                dark_mode=dark_mode
            )
            
            results_html = html.Div([
                dcc.Graph(
                    figure=fig,
                    config={'displayModeBar': True}
                )
            ])
            
            # Store results
            results_data = {'type': 'stft'}
            
            status_msg = create_status_message("STFT analysis completed", "success", dark_mode)
            
        elif analysis_type == 'multi_phase':
            # Multi-phase analysis
            # For now, we'll simulate having multiple phases
            # In a real app, you would load data for all phases
            
            # Create simulated phase data (in reality, load from database)
            phase_data = {}
            for phase in multi_phase_selection:
                # Create slightly different data for each phase for demonstration
                phase_copy = dict(data)
                
                # Add some phase shift and amplitude variation
                phase_int = int(phase)
                phase_shift = (phase_int - 1) * np.pi / 3  # 60 degrees between phases
                amplitude_factor = 0.9 + phase_int * 0.1  # Slight amplitude differences
                
                phase_copy['current'] = amplitude_factor * data['current'] * np.cos(phase_shift)
                phase_data[phase] = phase_copy
            
            # Phase map for display labels
            phase_map = {'1': 'A', '2': 'B', '3': 'C'}
            
            # Perform selected analysis
            if multi_phase_type == 'thd':
                # Harmonics comparison
                multi_phase_result = signal_processor.analyze_multi_phase(
                    phase_data,
                    analysis_type='thd',
                    phases=multi_phase_selection
                )
                
                # Store multi-phase data for reuse
                multi_phase_data_result = convert_to_json_serializable(multi_phase_result)
                
                # Create multi-phase harmonics plot
                fig = plot_generator.create_multi_phase_harmonic_plot(
                    multi_phase_result,
                    plot_style=multi_phase_plot_style,
                    dark_mode=dark_mode,
                    phase_map=phase_map
                )
                
                # Create comparison metrics
                phase_thds = {}
                phase_fundamentals = {}
                for phase, data in multi_phase_result['phase_results'].items():
                    phase_thds[phase] = data['thd']
                    phase_fundamentals[phase] = {
                        'freq': data['fundamental_freq'],
                        'magnitude': data['fundamental_magnitude']
                    }
                
                # Create metrics card
                metrics_card = html.Div([
                    html.H5("Comparative Metrics", style={'marginBottom': '15px'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Phase"),
                            html.Th("THD (%)"),
                            html.Th("Fundamental Freq (Hz)"),
                            html.Th("Fundamental Magnitude")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"Phase {phase_map.get(phase, phase)}"),
                                html.Td(f"{thd:.2f}%"),
                                html.Td(f"{phase_fundamentals[phase]['freq']:.2f}"),
                                html.Td(f"{phase_fundamentals[phase]['magnitude']:.5f}")
                            ]) for phase, thd in phase_thds.items()
                        ])
                    ], style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'marginBottom': '20px'
                    })
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                # Create harmonic details table
                harmonics_table = html.Div([
                    html.H5("Individual Harmonic Comparison", style={'marginBottom': '15px'}),
                    html.Div([  # Scrollable container
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Harmonic"),
                            ] + [html.Th(f"Phase {phase_map.get(phase, phase)}") for phase in multi_phase_selection])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(f"H{i+1}"),
                                ] + [
                                    html.Td(f"{phase_results['harmonics'][i]['magnitude']:.5f}")
                                    for phase, phase_results in multi_phase_result['phase_results'].items()
                                    if phase in multi_phase_selection
                                ]) for i in range(10)  # Show first 10 harmonics
                            ])
                        ], style={
                            'width': '100%',
                            'borderCollapse': 'collapse'
                        })
                    ], style={
                        'maxHeight': '300px',
                        'overflowY': 'auto'
                    })
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                results_html = html.Div([
                    metrics_card,
                    harmonics_table,
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
                
            elif multi_phase_type == 'fft':
                # FFT comparison
                multi_phase_result = signal_processor.analyze_multi_phase(
                    phase_data,
                    analysis_type='fft',
                    phases=multi_phase_selection
                )
                
                # Store multi-phase data for reuse
                multi_phase_data_result = convert_to_json_serializable(multi_phase_result)
                
                # Create multi-phase FFT plot
                fig = plot_generator.create_multi_phase_fft_plot(
                    multi_phase_result,
                    plot_style=multi_phase_plot_style,
                    dark_mode=dark_mode,
                    phase_map=phase_map
                )
                
                results_html = html.Div([
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
                
            elif multi_phase_type == 'correlation':
                # Phase correlation analysis
                multi_phase_result = signal_processor.analyze_multi_phase(
                    phase_data,
                    analysis_type='correlation',
                    phases=multi_phase_selection
                )
                
                # Store multi-phase data for reuse
                multi_phase_data_result = convert_to_json_serializable(multi_phase_result)
                
                # Create correlation plot
                fig = plot_generator.create_multi_phase_correlation_plot(
                    multi_phase_result,
                    dark_mode=dark_mode,
                    phase_map=phase_map
                )
                
                # Extract phase angle data for display
                correlation_results = multi_phase_result['correlation_results']
                phase_angles = correlation_results['phase_angles']
                reference_phase = correlation_results['reference_phase']
                
                # Create phase angle table
                angle_table = html.Div([
                    html.H5("Phase Angle Measurements", style={'marginBottom': '15px'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Phase"),
                            html.Th("Angle (degrees)"),
                            html.Th("Relative to")
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"Phase {phase_map.get(phase, phase)}"),
                                html.Td(f"{angle:.2f}°"),
                                html.Td(f"Phase {phase_map.get(reference_phase, reference_phase)}")
                            ]) for phase, angle in phase_angles.items()
                        ])
                    ], style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'marginBottom': '20px'
                    })
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                results_html = html.Div([
                    angle_table,
                    dcc.Graph(
                        figure=fig,
                        config={'displayModeBar': True}
                    )
                ])
                
            else:  # time or other types
                # Time domain comparison
                multi_phase_result = signal_processor.analyze_multi_phase(
                    phase_data,
                    analysis_type='time',
                    phases=multi_phase_selection
                )
                
                # Store multi-phase data for reuse
                multi_phase_data_result = convert_to_json_serializable(multi_phase_result)
                
                # For now, just display metrics
                phase_metrics = {}
                for phase, data in multi_phase_result['phase_results'].items():
                    phase_metrics[phase] = data['metrics']
                
                # Create metrics card
                metrics_card = html.Div([
                    html.H5("Phase Comparison Metrics", style={'marginBottom': '15px'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Metric"),
                        ] + [html.Th(f"Phase {phase_map.get(phase, phase)}") for phase in multi_phase_selection])),
                        html.Tbody([
                            html.Tr([
                                html.Td("RMS Current"),
                            ] + [
                                html.Td(f"{metrics['rms']:.4f} A")
                                for phase, metrics in phase_metrics.items()
                                if phase in multi_phase_selection
                            ]),
                            html.Tr([
                                html.Td("Peak-to-Peak"),
                            ] + [
                                html.Td(f"{metrics['peak_to_peak']:.4f} A")
                                for phase, metrics in phase_metrics.items()
                                if phase in multi_phase_selection
                            ]),
                            html.Tr([
                                html.Td("Crest Factor"),
                            ] + [
                                html.Td(f"{metrics['crest_factor']:.4f}")
                                for phase, metrics in phase_metrics.items()
                                if phase in multi_phase_selection
                            ]),
                            html.Tr([
                                html.Td("Form Factor"),
                            ] + [
                                html.Td(f"{metrics['form_factor']:.4f}")
                                for phase, metrics in phase_metrics.items()
                                if phase in multi_phase_selection
                            ])
                        ])
                    ], style={
                        'width': '100%',
                        'borderCollapse': 'collapse',
                        'marginBottom': '20px'
                    })
                ], style={
                    'backgroundColor': DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    'borderRadius': '5px',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
                
                # Simple placeholder for now
                results_html = metrics_card
            
            # Store results
            results_data = {'type': 'multi_phase', 'analysis_type': multi_phase_type}
            
            status_msg = create_status_message(f"Multi-phase {multi_phase_type} analysis completed", "success", dark_mode)
            
        else:
            # Unknown analysis type
            results_html = html.Div(f"Analysis type {analysis_type} not implemented")
            results_data = None
            status_msg = create_status_message(
                f"Analysis type {analysis_type} not implemented", "warning", dark_mode
            )
        
        return results_html, results_data, harmonics_data_result, wavelet_data_result, power_quality_data_result, transient_data_result, multi_phase_data_result, status_msg, {'opacity': '1'}
        
    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return html.Div(f"Analysis error: {str(e)}"), None, None, None, None, None, None, create_status_message(error_msg, "danger", dark_mode), {'opacity': '1'}

# Start the app
if __name__ == '__main__':
    app.run(debug=True)