import plotly.graph_objs as go
import plotly.subplots as sp
import numpy as np
from scipy import signal
import logging
import plotly.express as px
from config import DARK_MODE_STYLES, LIGHT_MODE_STYLES

logger = logging.getLogger(__name__)

class PlotGenerator:
    def create_time_domain_plot(self, data, show_mean=True, show_rms=True, show_envelope=False, dark_mode=False):
        """Create time domain plot from data"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return go.Figure()
            
        # Create figure
        fig = go.Figure()
        
        # Add main current trace
        fig.add_trace(go.Scatter(
            x=data['time'],
            y=data['current'],
            mode='lines',
            name='Current',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        ))
        
        # Add mean line if requested
        if show_mean:
            mean_value = np.mean(data['current'])
            fig.add_trace(go.Scatter(
                x=[data['time'][0], data['time'][-1]],
                y=[mean_value, mean_value],
                mode='lines',
                name='Mean',
                line=dict(color='red', width=1, dash='dash')
            ))
        
        # Add RMS line if requested
        if show_rms:
            rms_value = np.sqrt(np.mean(np.square(data['current'])))
            fig.add_trace(go.Scatter(
                x=[data['time'][0], data['time'][-1]],
                y=[rms_value, rms_value],
                mode='lines',
                name='RMS',
                line=dict(color='green', width=1, dash='dot')
            ))
            
            # Add negative RMS line
            fig.add_trace(go.Scatter(
                x=[data['time'][0], data['time'][-1]],
                y=[-rms_value, -rms_value],
                mode='lines',
                name='-RMS',
                line=dict(color='green', width=1, dash='dot'),
                showlegend=False
            ))
        
        # Add envelope if requested
        if show_envelope:
            # Compute analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(data['current'])
            envelope = np.abs(analytic_signal)
            
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=envelope,
                mode='lines',
                name='Envelope',
                line=dict(color='purple', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=-envelope,
                mode='lines',
                name='-Envelope',
                line=dict(color='purple', width=1),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='Time Domain Analysis',
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            template='plotly_dark' if dark_mode else 'plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_power_quality_index_plot(self, pqi_data, dark_mode=False):
        """Create Power Quality Index visualization"""
        if pqi_data is None or 'pqi' not in pqi_data:
            return go.Figure()
        
        # Create figure with subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Overall Power Quality Index', 
                'Component Indices',
                'Raw Metrics',
                'Quality Assessment'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "table"}, {"type": "domain"}]
            ]
        )
    
        # 1. Gauge indicator for overall PQI
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=pqi_data['pqi'],
                title={'text': "Power Quality Index"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self._get_quality_color(pqi_data['pqi'])},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "orange"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 90], 'color': "lightgreen"},
                        {'range': [90, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                },
                delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}}
            ),
            row=1, col=1
        )
    
        # 2. Component indices bar chart
        components = pqi_data['components']
        fig.add_trace(
            go.Bar(
                x=['Crest Factor', 'Form Factor', 'THD', 'Transients'],
                y=[
                    components['crest_factor_index'],
                    components['form_factor_index'],
                    components['thd_index'],
                    components['transient_index']
                ],
                marker_color=[
                    self._get_quality_color(components['crest_factor_index']),
                    self._get_quality_color(components['form_factor_index']),
                    self._get_quality_color(components['thd_index']),
                    self._get_quality_color(components['transient_index'])
                ]
            ),
            row=1, col=2
        )
    
        # 3. Raw metrics table
        raw_metrics = pqi_data['raw_metrics']
        metrics_table = go.Table(
            header=dict(
                values=['Metric', 'Value', 'Ideal', 'Score'],
                font=dict(size=12, color='white' if dark_mode else 'black'),
                fill_color='rgba(0, 102, 204, 0.8)'
            ),
            cells=dict(
                values=[
                    ['Crest Factor', 'Form Factor', 'THD (%)', 'Transients'],
                    [
                        f"{raw_metrics['crest_factor']:.3f}",
                        f"{raw_metrics['form_factor']:.3f}",
                        f"{raw_metrics['thd']:.2f}",
                        f"{raw_metrics['transient_count']}"
                    ],
                    ['1.414', '1.11', '0', '0'],
                    [
                        f"{components['crest_factor_index']:.1f}",
                        f"{components['form_factor_index']:.1f}",
                        f"{components['thd_index']:.1f}",
                        f"{components['transient_index']:.1f}"
                    ]
                ],
                font=dict(size=11),
                fill_color=[
                    'rgba(242, 242, 242, 1)' if not dark_mode else 'rgba(50, 50, 50, 1)',
                    'rgba(242, 242, 242, 1)' if not dark_mode else 'rgba(50, 50, 50, 1)',
                    'rgba(242, 242, 242, 1)' if not dark_mode else 'rgba(50, 50, 50, 1)',
                    [
                        self._get_quality_color_with_alpha(components['crest_factor_index']),
                        self._get_quality_color_with_alpha(components['form_factor_index']),
                        self._get_quality_color_with_alpha(components['thd_index']),
                        self._get_quality_color_with_alpha(components['transient_index'])
                    ]
                ]
            )
        )
        fig.add_trace(metrics_table, row=2, col=1)
    
        # 4. Quality assessment pie chart
        # Fix: Change title position to a valid value
        fig.add_trace(
            go.Pie(
                labels=['Crest Factor', 'Form Factor', 'THD', 'Transients'],
                values=[0.25, 0.20, 0.40, 0.15],  # Component weights
                textinfo='label+percent',
                marker=dict(
                    colors=[
                        self._get_quality_color(components['crest_factor_index']),
                        self._get_quality_color(components['form_factor_index']),
                        self._get_quality_color(components['thd_index']),
                        self._get_quality_color(components['transient_index'])
                    ]
                ),
                hole=0.4,
                title=dict(
                    text=f"Quality: {pqi_data['quality_level']}",
                    position="middle center",  # Changed from "middle" to "middle center"
                    font=dict(size=14)
                )
            ),
            row=2, col=2
        )
    
        # Update layout
        fig.update_layout(
            title='Power Quality Index Analysis',
            height=800
        )
    
        # Update y-axis titles
        fig.update_yaxes(title_text='Score (0-100)', row=1, col=2)
    
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
    
        return fig

    def _get_quality_color(self, index):
        """Get color based on quality index"""
        if index >= 90:
            return 'green'
        elif index >= 80:
            return 'lightgreen'
        elif index >= 60:
            return 'yellow'
        elif index >= 40:
            return 'orange'
        else:
            return 'red'

    def _get_quality_color_with_alpha(self, index):
        """Get color with alpha based on quality index"""
        if index >= 90:
            return 'rgba(0, 128, 0, 0.3)'  # green
        elif index >= 80:
            return 'rgba(144, 238, 144, 0.3)'  # lightgreen
        elif index >= 60:
            return 'rgba(255, 255, 0, 0.3)'  # yellow
        elif index >= 40:
            return 'rgba(255, 165, 0, 0.3)'  # orange
        else:
            return 'rgba(255, 0, 0, 0.3)'  # red

    def create_frequency_domain_plot(self, fft_data, scale='db', max_freq=500, dark_mode=False):
        """Create frequency domain plot from FFT data"""
        if fft_data is None or 'freq' not in fft_data or len(fft_data['freq']) == 0:
            return go.Figure()
            
        # Create figure
        fig = go.Figure()
        
        # Choose magnitude or dB scale based on parameter
        if scale == 'db':
            y_data = fft_data['magnitude_db']
            y_title = 'Magnitude (dB)'
        elif scale == 'log-log':
            y_data = fft_data['magnitude']
            y_title = 'Magnitude (log)'
        else:  # linear
            y_data = fft_data['magnitude']
            y_title = 'Magnitude'
        
        # Main trace
        fig.add_trace(go.Scatter(
            x=fft_data['freq'],
            y=y_data,
            mode='lines',
            name='Magnitude Spectrum',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1.5)
        ))
        
        # Update layout
        fig.update_layout(
            title='Frequency Domain Analysis',
            xaxis_title='Frequency (Hz)',
            yaxis_title=y_title,
            xaxis_range=[0, max_freq],  # Limit frequency range for better visibility
            template='plotly_dark' if dark_mode else 'plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Set log scales if needed
        if scale == 'log-log':
            fig.update_layout(
                xaxis_type='log',
                yaxis_type='log'
            )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_spectrogram_plot(self, spectrogram_data, colormap='viridis', dark_mode=False):
        """Create spectrogram plot"""
        if spectrogram_data is None or 'power' not in spectrogram_data:
            return go.Figure()
            
        # Create figure using 3D surface for a different visualization
        fig = go.Figure()
        
        # Extract and reshape data
        power = spectrogram_data['power']
        times = spectrogram_data['time']
        freqs = spectrogram_data['freq']

         # Use contour plot instead of heatmap for distinction from STFT
        fig.add_trace(go.Contour(
            z=power,
            x=times,
            y=freqs,
            colorscale=colormap,
            contours=dict(
                start=-80,
                end=0,
                size=5,
                showlabels=True,
                labelfont=dict(
                    color='white' if dark_mode else 'black',
                    size=10,
                )
            ),
            colorbar=dict(title='Power (dB)')
        ))
        
        # Update layout
        fig.update_layout(
            title='Spectrogram Analysis (Contour View)',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            yaxis_range=[0, min(500, max(spectrogram_data['freq']))],  # Limit to 0-500 Hz for better visibility
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_interharmonics_plot(self, interharmonics_data, dark_mode=False):
        """Create interharmonics analysis plot"""
        if interharmonics_data is None or 'interharmonic_groups' not in interharmonics_data:
            return go.Figure()
            
        # Extract data
        groups = interharmonics_data['interharmonic_groups']
        fundamental_freq = interharmonics_data['fundamental_freq']
        
        # Create data for bar chart
        x = [g['group'] for g in groups]
        y = [g['magnitude'] for g in groups]
        frequencies = [g['frequency'] for g in groups]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name='Magnitude',
            marker_color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
            text=[f"{freq:.1f} Hz" for freq in frequencies],
            textposition='inside',
            textfont=dict(
                color='white' if dark_mode else 'black',
                size=10
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Interharmonics Analysis - TID: {interharmonics_data['tid']:.2f}%",
            xaxis_title='Interharmonic Group',
            yaxis_title='Magnitude',
            xaxis=dict(
                tickmode='array',
                tickvals=x,
                ticktext=[f"Group {g}" for g in x]
            ),
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_cepstrum_plot(self, cepstrum_data, dark_mode=False):
        """Create cepstrum analysis plot"""
        if cepstrum_data is None or 'cepstrum' not in cepstrum_data:
            return go.Figure()
            
        # Extract data
        quefrency = cepstrum_data['quefrency']
        cepstrum = cepstrum_data['cepstrum']
        peaks = cepstrum_data['peaks']
        
        # Create figure
        fig = go.Figure()
        
        # Add cepstrum trace
        fig.add_trace(go.Scatter(
            x=quefrency,
            y=cepstrum,
            mode='lines',
            name='Cepstrum',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        ))
        
        # Mark peaks
        for i, peak in enumerate(peaks):
            fig.add_trace(go.Scatter(
                x=[peak['quefrency']],
                y=[peak['amplitude']],
                mode='markers+text',
                name=f"Peak {i+1}",
                text=[f"{peak['frequency']:.1f} Hz"],
                textposition="top center",
                marker=dict(
                    size=10,
                    color='red',
                    symbol='circle'
                ),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='Cepstrum Analysis',
            xaxis_title='Quefrency (s)',
            yaxis_title='Amplitude',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig    

    def create_stft_plot(self, stft_data, colormap='viridis', dark_mode=False):
        """Create STFT plot"""
        if stft_data is None or 'magnitude_db' not in stft_data:
            return go.Figure()
            
        # Create figure using heatmap
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(go.Heatmap(
            z=stft_data['magnitude_db'],
            x=stft_data['time'],
            y=stft_data['freq'],
            colorscale=colormap,
            colorbar=dict(title='Magnitude (dB)')
        ))
        
        # Update layout
        fig.update_layout(
            title='Short-Time Fourier Transform (STFT)',
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            yaxis_range=[0, min(500, max(stft_data['freq']))],  # Limit to 0-500 Hz for better visibility
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig    
    
    def create_transient_plot(self, transient_data, dark_mode=False):
        """Create transients analysis plot"""
        if transient_data is None or 'type' not in transient_data or transient_data['type'] != 'transients':
            return go.Figure()
        
        # Extract data
        time = transient_data['time']
        current = transient_data['current']
        events = transient_data['events']
        threshold = transient_data['threshold']
    
        # Create figure
        fig = go.Figure()
    
        # Add current trace
        fig.add_trace(go.Scatter(
            x=time,
            y=current,
            mode='lines',
            name='Current',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        ))
    
        # Add threshold lines
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[threshold, threshold],
            mode='lines',
            name='Upper Threshold',
            line=dict(color='red', width=1, dash='dash')
        ))
    
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[-threshold, -threshold],
            mode='lines',
            name='Lower Threshold',
            line=dict(color='red', width=1, dash='dash')
        ))
    
        # Mark detected events with different colors based on classification
        colors = {
            'Impulse': 'rgba(255, 0, 0, 0.3)',  # Red
            'Oscillatory': 'rgba(0, 255, 0, 0.3)',  # Green
            'Swell': 'rgba(0, 0, 255, 0.3)',  # Blue
            'Dip': 'rgba(255, 165, 0, 0.3)'   # Orange
        }
    
        # Create a list of unique event types for the legend
        event_types = set()
    
        for i, event in enumerate(events):
            event_class = event.get('class', 'Unknown')
            event_types.add(event_class)
        
            # Mark start and end of event with rectangle
            fig.add_shape(
                type="rect",
                x0=event['start_time'],
                x1=event['end_time'],
                y0=min(current) - 0.1 * (max(current) - min(current)),
                y1=max(current) + 0.1 * (max(current) - min(current)),
                line=dict(
                    color="rgba(0, 0, 0, 0)",
                    width=1,
                    dash="dot",
                ),
                fillcolor=colors.get(event_class, 'rgba(128, 128, 128, 0.3)'),
                layer="below"
            )
        
            # Add annotation for event number
            fig.add_annotation(
                x=event['peak_time'],
                y=event['peak_value'],
                text=f"Event {i+1}<br>{event_class}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40 if event['peak_value'] > 0 else 40
            )
    
        # Add legend for event types
        for event_type in event_types:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colors.get(event_type, 'rgba(128, 128, 128, 0.3)')),
                name=f"{event_type} Event"
            ))
    
        # Update layout
        fig.update_layout(
            title=f"Transient Analysis - {len(events)} Event(s) Detected",
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
    
        return fig
    
    def create_multi_phase_correlation_plot(self, correlation_data, dark_mode=False, phase_map=None):
        """Create multi-phase correlation plot"""
        if correlation_data is None or 'type' not in correlation_data or correlation_data['type'] != 'correlation':
            return go.Figure()
            
        # Extract data
        results = correlation_data['correlation_results']
        correlation_matrix = results['correlation_matrix']
        phase_angles = results['phase_angles']
        phases = list(phase_angles.keys())
        
        # Create heatmap data
        x_labels = phases
        y_labels = phases
        z_values = []
        
        for phase1 in y_labels:
            row = []
            for phase2 in x_labels:
                row.append(correlation_matrix[phase1][phase2])
            z_values.append(row)
        
        # Create correlation heatmap
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation'),
            text=[[f"{z:.3f}" for z in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size":12}
        ))
        
        # Create phase angle plot
        angles_fig = go.Figure()
        
        # Convert to polar coordinates
        reference_phase = results['reference_phase']
        r = [1] * len(phases)
        theta = [np.radians(phase_angles[phase]) for phase in phases]
        
        # Add angle markers
        angles_fig.add_trace(go.Scatterpolar(
            r=r,
            theta=[np.degrees(t) for t in theta],
            mode='markers+text',
            marker=dict(size=12),
            text=phases,
            textposition="top center",
            name='Phases'
        ))
        
        # Add connecting lines
        angles_fig.add_trace(go.Scatterpolar(
            r=[0] + r,
            theta=[0] + [np.degrees(t) for t in theta],
            mode='lines',
            line=dict(dash='dot'),
            showlegend=False
        ))
        
        # Update layout
        angles_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(
                    visible=True,
                    tickmode='array',
                    tickvals=[0, 90, 180, -90],
                    ticktext=['0°', '90°', '180°', '-90°']
                )
            ),
            showlegend=False
        )
        
        # Create combined figure with subplots
        combined_fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=('Phase Correlation Matrix', 'Phase Angle Diagram'),
            specs=[[{"type": "heatmap"}, {"type": "polar"}]],
            column_widths=[0.6, 0.4]
        )
        
        # Add heatmap to subplot
        combined_fig.add_trace(go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='Viridis',
            zmin=-1, zmax=1,
            colorbar=dict(title='Correlation'),
            text=[[f"{z:.3f}" for z in row] for row in z_values],
            texttemplate="%{text}",
            textfont={"size":12}
        ), row=1, col=1)
        
        # Add polar plot to subplot
        combined_fig.add_trace(go.Scatterpolar(
            r=r,
            theta=[np.degrees(t) for t in theta],
            mode='markers+text',
            marker=dict(size=12),
            text=phases,
            textposition="top center",
            name='Phases'
        ), row=1, col=2)
        
        # Add connecting lines
        combined_fig.add_trace(go.Scatterpolar(
            r=[0] + r,
            theta=[0] + [np.degrees(t) for t in theta],
            mode='lines',
            line=dict(dash='dot'),
            showlegend=False
        ), row=1, col=2)
        
        # Update layout
        combined_fig.update_layout(
            title='Multi-Phase Correlation Analysis',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            height=500
        )
        
        # Configure polar subplot
        combined_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 1]),
                angularaxis=dict(
                    visible=True,
                    tickmode='array',
                    tickvals=[0, 90, 180, -90],
                    ticktext=['0°', '90°', '180°', '-90°']
                )
            )
        )
        
        # Apply theme
        combined_fig = self._apply_theme(combined_fig, dark_mode)
        
        return combined_fig    

    def create_waveform_distortion_plot(self, distortion_data, dark_mode=False):
        """Create waveform distortion analysis plot"""
        if distortion_data is None or 'thd' not in distortion_data:
            return go.Figure()
            
        # Create figure with subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distortion Parameters',
                'THD Breakdown',
                'Top 10 Harmonics',
                'Harmonics by Type'
            ),
            specs=[
                [{"type": "table"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. Distortion Parameters Table
        parameters = [
            ['THD (%)', f"{distortion_data['thd']:.2f}"],
            ['K-Factor', f"{distortion_data['k_factor']:.2f}"],
            ['Crest Factor', f"{distortion_data['crest_factor']:.2f}"],
            ['Form Factor', f"{distortion_data['form_factor']:.2f}"],
            ['Transformer Derating', f"{distortion_data['transformer_derating_factor']:.2f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Parameter', 'Value'],
                    font=dict(size=12, color='white' if dark_mode else 'black'),
                    fill_color=DARK_MODE_STYLES['header_bg'] if dark_mode else LIGHT_MODE_STYLES['header_bg'],
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*parameters)),
                    font=dict(size=12),
                    fill_color=DARK_MODE_STYLES['card_bg'] if dark_mode else LIGHT_MODE_STYLES['card_bg'],
                    align='left'
                )
            ),
            row=1, col=1
        )
        
        # 2. THD Breakdown Pie Chart
        thd_breakdown = [
            distortion_data['even_thd'],
            distortion_data['odd_thd'] - distortion_data['triplen_thd'],  # Non-triplen odd harmonics
            distortion_data['triplen_thd']
        ]
        
        fig.add_trace(
            go.Pie(
                labels=['Even Harmonics', 'Odd Harmonics (Non-Triplen)', 'Triplen Harmonics'],
                values=thd_breakdown,
                hole=0.4,
                textinfo='percent+label',
                marker_colors=['rgba(55, 128, 191, 0.8)', 'rgba(219, 64, 82, 0.8)', 'rgba(50, 171, 96, 0.8)']
            ),
            row=1, col=2
        )
        
        # 3. Top 10 Harmonics Bar Chart
        harmonics = distortion_data['harmonics'][1:11]  # Skip fundamental, get next 10
        
        fig.add_trace(
            go.Bar(
                x=[f"H{h['harmonic']}" for h in harmonics],
                y=[h['magnitude'] for h in harmonics],
                text=[f"{h['magnitude']:.4f}" for h in harmonics],
                textposition='outside',
                marker_color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary']
            ),
            row=2, col=1
        )
        
        # 4. Harmonics by Type
        even_data = [distortion_data['even_thd']]
        odd_data = [distortion_data['odd_thd'] - distortion_data['triplen_thd']]
        triplen_data = [distortion_data['triplen_thd']]
        
        fig.add_trace(
            go.Bar(
                x=['Even THD'],
                y=even_data,
                name='Even',
                text=[f"{val:.2f}%" for val in even_data],
                textposition='outside',
                marker_color='rgba(55, 128, 191, 0.8)'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=['Odd THD (Non-Triplen)'],
                y=odd_data,
                name='Odd (Non-Triplen)',
                text=[f"{val:.2f}%" for val in odd_data],
                textposition='outside',
                marker_color='rgba(219, 64, 82, 0.8)'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=['Triplen THD'],
                y=triplen_data,
                name='Triplen',
                text=[f"{val:.2f}%" for val in triplen_data],
                textposition='outside',
                marker_color='rgba(50, 171, 96, 0.8)'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Waveform Distortion Analysis',
            template='plotly_dark' if dark_mode else 'plotly_white',
            height=800,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text='Harmonic', row=2, col=1)
        fig.update_yaxes(title_text='Magnitude', row=2, col=1)
        
        fig.update_xaxes(title_text='Type', row=2, col=2)
        fig.update_yaxes(title_text='THD (%)', row=2, col=2)
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig

    def create_harmonics_plot(self, harmonics_data, num_harmonics=15, dark_mode=False):
        """Create harmonics bar chart"""
        if harmonics_data is None or 'harmonics' not in harmonics_data:
            return go.Figure()
            
        # Extract data
        harmonics = harmonics_data['harmonics'][:num_harmonics]
        
        # Create data for bar chart
        x = [h['harmonic'] for h in harmonics]
        y = [h['magnitude'] for h in harmonics]
        frequencies = [h['frequency'] for h in harmonics]
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name='Magnitude',
            marker_color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'],
            text=[f"{freq:.1f} Hz" for freq in frequencies],
            textposition='inside',
            textfont=dict(
                color='white' if dark_mode else 'black',
                size=10
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Harmonic Spectrum - THD: {harmonics_data['thd']:.2f}%",
            xaxis_title='Harmonic',
            yaxis_title='Magnitude',
            xaxis=dict(
                tickmode='array',
                tickvals=x,
                ticktext=[f"H{h}" for h in x]
            ),
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_harmonic_components_plot(self, harmonics_data, num_harmonics=5, dark_mode=False):
        """Create individual harmonic components plot"""
        if harmonics_data is None or 'harmonics' not in harmonics_data:
            return go.Figure()
            
        # Extract data for the first n harmonics
        harmonics = harmonics_data['harmonics'][:num_harmonics]
        fund_freq = harmonics_data['fundamental_freq']
        
        # Create time array (one cycle of fundamental)
        t = np.linspace(0, 1/fund_freq, 1000)
        
        # Create figure
        fig = go.Figure()
        
        # Create waveforms for each harmonic
        colors = px.colors.qualitative.Plotly
        for i, h in enumerate(harmonics):
            magnitude = h['magnitude']
            frequency = h['frequency']
            phase = h['phase']
            
            # Generate waveform
            y = magnitude * np.sin(2 * np.pi * frequency * t + phase)
            
            # Add trace
            fig.add_trace(go.Scatter(
                x=t,
                y=y,
                mode='lines',
                name=f"H{i+1} ({frequency:.1f} Hz)",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title='Individual Harmonic Components',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_signal_reconstruction_plot(self, harmonics_data, selected_harmonics=None, dark_mode=False):
        """Create signal reconstruction plot from selected harmonics"""
        if harmonics_data is None or 'harmonics' not in harmonics_data:
            return go.Figure()
            
        # If selected_harmonics is None, use all harmonics
        if selected_harmonics is None:
            selected_harmonics = [h+1 for h in range(len(harmonics_data['harmonics']))]
        
        # Extract data
        harmonics = harmonics_data['harmonics']
        fund_freq = harmonics_data['fundamental_freq']
        
        # Create time array (one cycle of fundamental)
        t = np.linspace(0, 1/fund_freq, 1000)
        
        # Create original signal (all harmonics)
        y_original = np.zeros_like(t)
        for h in harmonics:
            magnitude = h['magnitude']
            frequency = h['frequency']
            phase = h['phase']
            y_original += magnitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # Create reconstructed signal (selected harmonics)
        y_reconstructed = np.zeros_like(t)
        for h_num in selected_harmonics:
            if h_num <= len(harmonics):
                h = harmonics[h_num-1]
                magnitude = h['magnitude']
                frequency = h['frequency']
                phase = h['phase']
                y_reconstructed += magnitude * np.sin(2 * np.pi * frequency * t + phase)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=t,
            y=y_original,
            mode='lines',
            name='Original',
            line=dict(color='gray', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=t,
            y=y_reconstructed,
            mode='lines',
            name='Reconstructed',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title='Signal Reconstruction from Selected Harmonics',
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_wavelet_plot(self, wavelet_data, dark_mode=False):
        """Create wavelet decomposition plot"""
        if wavelet_data is None or 'approximation' not in wavelet_data:
            return go.Figure()
        
        # Extract data
        time = wavelet_data['time']
        approximation = wavelet_data['approximation']
        details = wavelet_data['details']
        level = wavelet_data['level']
        
        # Create figure with subplots
        fig = sp.make_subplots(
            rows=level+1, 
            cols=1,
            subplot_titles=['Original'] + [f'Level {i+1}' for i in range(level)],
            vertical_spacing=0.05
        )
        
        # Add original (approximation) trace
        fig.add_trace(
            go.Scatter(
                x=time, 
                y=approximation,
                mode='lines',
                name='Original',
                line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'])
            ),
            row=1, col=1
        )
        
        # Add detail traces
        colors = px.colors.qualitative.Plotly
        for i, detail in enumerate(details):
            fig.add_trace(
                go.Scatter(
                    x=time, 
                    y=detail,
                    mode='lines',
                    name=f'Level {i+1}',
                    line=dict(color=colors[(i+1) % len(colors)])
                ),
                row=i+2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Wavelet Analysis ({wavelet_data["wavelet_type"]}) - {level} levels',
            template='plotly_dark' if dark_mode else 'plotly_white',
            height=200*(level+1),
            showlegend=True,
            hovermode='closest'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text='Current (A)', row=1, col=1)
        for i in range(level):
            fig.update_yaxes(title_text='Amplitude', row=i+2, col=1)
        
        # Update x-axis labels (only show on bottom plot)
        for i in range(level):
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
        fig.update_xaxes(title_text='Time (s)', row=level+1, col=1)
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_flicker_plot(self, flicker_data, dark_mode=False):
        """Create flicker analysis plot"""
        if flicker_data is None or 'type' not in flicker_data or flicker_data['type'] != 'flicker':
            return go.Figure()
            
        # Extract data
        time = flicker_data['time']
        current = flicker_data['current']
        envelope = flicker_data['envelope']
        
        # Create figure
        fig = go.Figure()
        
        # Add current trace
        fig.add_trace(go.Scatter(
            x=time,
            y=current,
            mode='lines',
            name='Current',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        ))
        
        # Add envelope trace
        fig.add_trace(go.Scatter(
            x=time,
            y=envelope,
            mode='lines',
            name='Envelope',
            line=dict(color='red', width=1)
        ))
        
        # Add negative envelope trace
        fig.add_trace(go.Scatter(
            x=time,
            y=-envelope,
            mode='lines',
            name='-Envelope',
            line=dict(color='red', width=1),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Flicker Analysis - NVL36 Phase 1",
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_transients_plot(self, transients_data, dark_mode=False):
        """Create transients analysis plot"""
        if transients_data is None or 'type' not in transients_data or transients_data['type'] != 'transients':
            return go.Figure()
            
        # Extract data
        time = transients_data['time']
        current = transients_data['current']
        events = transients_data['events']
        baseline_rms = transients_data['baseline_rms']
        
        # Create figure
        fig = go.Figure()
        
        # Add current trace
        fig.add_trace(go.Scatter(
            x=time,
            y=current,
            mode='lines',
            name='Current',
            line=dict(color=DARK_MODE_STYLES['button_primary'] if dark_mode else LIGHT_MODE_STYLES['button_primary'], width=1)
        ))
        
        # Add baseline trace
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[0, 0],
            mode='lines',
            name='Baseline',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        # Mark detected events with vertical lines
        for event in events:
            fig.add_vline(
                x=event['start_time'],
                line_width=1,
                line_dash="dash",
                line_color="red"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Transient Analysis - NVL36 Phase 3",
            xaxis_title='Time (s)',
            yaxis_title='Current (A)',
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest'
        )
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def create_multi_phase_harmonic_plot(self, multi_phase_data, plot_style='overlay', dark_mode=False, phase_map=None):
        """Create multi-phase harmonics comparison plot"""
        if multi_phase_data is None or 'type' not in multi_phase_data or multi_phase_data['type'] != 'thd':
            return go.Figure()
            
        # Extract data
        phase_results = multi_phase_data['phase_results']
        phases = list(phase_results.keys())
        
        # Use provided phase_map or default to phase number
        if phase_map is None:
            phase_map = {phase: phase for phase in phases}
    
        if plot_style == 'overlay':
            # Create figure
            fig = go.Figure()
        
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
            
                # Extract harmonic data
                harmonics = phase_data['harmonics']
                x = [h['harmonic'] for h in harmonics[:10]]  # First 10 harmonics
                y = [h['magnitude'] for h in harmonics[:10]]
            
                # Add bar chart
                fig.add_trace(go.Bar(
                    x=x,
                    y=y,
                    name=f'Phase {phase_map.get(phase, phase)}',
                    marker_color=colors[i % len(colors)]
                ))
        
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase Harmonic Comparison",
                xaxis_title='Harmonic',
                yaxis_title='Magnitude',
                barmode='group',
                template='plotly_dark' if dark_mode else 'plotly_white',
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode='closest'
            )
        else:  # Separate
            # Create figure with subplots
            fig = sp.make_subplots(
                rows=len(phases), 
                cols=1,
                subplot_titles=[f'Phase {phase_map.get(phase, phase)}' for phase in phases],
                vertical_spacing=0.1
            )
        
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
            
                # Extract harmonic data
                harmonics = phase_data['harmonics']
                x = [h['harmonic'] for h in harmonics[:10]]  # First 10 harmonics
                y = [h['magnitude'] for h in harmonics[:10]]
            
                # Add bar chart
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        name=f'Phase {phase_map.get(phase, phase)}',
                        marker_color=colors[i % len(colors)]
                    ),
                    row=i+1, col=1
                )
        
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase Harmonic Analysis",
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=300*len(phases),
                showlegend=True,
                hovermode='closest'
            )
        
            # Update y-axis title
            for i in range(len(phases)):
                fig.update_yaxes(title_text='Magnitude', row=i+1, col=1)
        
            # Update x-axis (only show on bottom plot)
            for i in range(len(phases)-1):
                fig.update_xaxes(showticklabels=False, row=i+1, col=1)
            fig.update_xaxes(title_text='Harmonic', row=len(phases), col=1)
    
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
    
        return fig
    
    def create_3d_harmonic_visualization(self, data, dark_mode=False):
        """Create 3D visualization of harmonics over time"""
        # Calculate FFT for segments of the signal
        sample_rate = 1 / data['sample_interval']
        segment_size = int(sample_rate / 10)  # 0.1 second segments
    
        if len(data['current']) < segment_size * 3:
            return go.Figure()  # Not enough data
    
        segments = []
        times = []
        for i in range(0, len(data['current']) - segment_size, segment_size):
            segment = data['current'][i:i+segment_size]
            times.append(data['time'][i])
        
            # Calculate FFT for this segment
            fft = np.abs(np.fft.rfft(segment))
            freq = np.fft.rfftfreq(len(segment), data['sample_interval'])
        
            # Store only frequency components up to 500 Hz
            cutoff = np.searchsorted(freq, 500)
            segments.append(fft[:cutoff])
    
        # Make all segments the same length
        min_length = min(len(s) for s in segments)
        segments = [s[:min_length] for s in segments]
        freq = freq[:min_length]
    
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=np.array(segments),
            x=freq,
            y=times,
            colorscale='Viridis'
        )])
    
        # Update layout
        fig.update_layout(
            title='3D Harmonic Spectrum Over Time',
            scene=dict(
                xaxis_title='Frequency (Hz)',
                yaxis_title='Time (s)',
                zaxis_title='Magnitude',
                xaxis=dict(range=[0, 500]),
            ),
            template='plotly_dark' if dark_mode else 'plotly_white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
    
        return fig

    def create_multi_phase_time_plot(self, multi_phase_data, plot_style='overlay', dark_mode=False, phase_map=None):
        """Create multi-phase time domain comparison plot"""
        if multi_phase_data is None or 'type' not in multi_phase_data or multi_phase_data['type'] != 'time':
            return go.Figure()
        
        # Extract data
        phase_results = multi_phase_data['phase_results']
        phases = list(phase_results.keys())
    
        # Use provided phase_map or default to phase number
        if phase_map is None:
            phase_map = {phase: phase for phase in phases}

        if plot_style == 'overlay':
            # Create figure
            fig = go.Figure()
    
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
        
                # Add time domain trace
                fig.add_trace(go.Scatter(
                    x=phase_data['time'],
                    y=phase_data['current'],
                    mode='lines',
                    name=f'Phase {phase_map.get(phase, phase)}',
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
    
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase Time Domain Comparison",
                xaxis_title='Time (s)',
                yaxis_title='Current (A)',
                template='plotly_dark' if dark_mode else 'plotly_white',
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode='closest'
            )
        else:  # Separate
            # Create figure with subplots
            fig = sp.make_subplots(
                rows=len(phases), 
                cols=1,
                subplot_titles=[f'Phase {phase_map.get(phase, phase)}' for phase in phases],
                vertical_spacing=0.1
            )
    
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
        
                # Add time domain trace
                fig.add_trace(
                    go.Scatter(
                        x=phase_data['time'],
                        y=phase_data['current'],
                        mode='lines',
                        name=f'Phase {phase_map.get(phase, phase)}',
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=i+1, col=1
                )
    
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase Time Domain Analysis",
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=300*len(phases),
                showlegend=True,
                hovermode='closest'
            )
    
            # Update y-axis title
            for i in range(len(phases)):
                fig.update_yaxes(title_text='Current (A)', row=i+1, col=1)
    
            # Update x-axis (only show on bottom plot)
            for i in range(len(phases)-1):
                fig.update_xaxes(showticklabels=False, row=i+1, col=1)
            fig.update_xaxes(title_text='Time (s)', row=len(phases), col=1)

        # Apply theme
        fig = self._apply_theme(fig, dark_mode)

        return fig
    
    def generate_pdf_report(self, analysis_results, report_title, included_sections):
        """Generate a PDF report with analysis results"""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
        import io
        from PIL import Image as PILImage
    
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
    
        # Add styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        normal_style = styles['Normal']
    
        # Add title
        elements.append(Paragraph(report_title, title_style))
        elements.append(Spacer(1, 12))
    
        # Add date and time
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {now}", normal_style))
        elements.append(Spacer(1, 24))
    
        # Add included sections
        if 'time' in included_sections and 'time_plot' in analysis_results:
            elements.append(Paragraph("Time Domain Analysis", heading_style))
            elements.append(Spacer(1, 12))
        
            # Convert plot to image
            plot_img = io.BytesIO()
            analysis_results['time_plot'].write_image(plot_img)
            plot_img.seek(0)
            img = PILImage.open(plot_img)
            img = img.resize((500, 300))
            img_path = io.BytesIO()
            img.save(img_path, format='PNG')
            img_path.seek(0)
        
            elements.append(Image(img_path))
            elements.append(Spacer(1, 12))
        
            # Add metrics
            if 'metrics' in analysis_results:
                metrics = analysis_results['metrics']
                data = [['Metric', 'Value']]
                for key, value in metrics.items():
                    if key in ['rms', 'peak_to_peak', 'crest_factor', 'form_factor']:
                        data.append([key, f"{value:.4f}"])
            
                elements.append(Table(data))
        
            elements.append(Spacer(1, 24))
    
        # Add more sections similarly
    
        # Build PDF
        doc.build(elements)
    
        buffer.seek(0)
        return buffer

    def create_symmetrical_components_plot(self, sym_comp_data, dark_mode=False):
        """Create symmetrical components visualization"""
        if sym_comp_data is None:
            return go.Figure()
        
        # Create figure with subplots
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sequence Components', 'Phasor Diagram'),
            specs=[[{"type": "bar"}, {"type": "polar"}]]
        )
    
        # Extract data
        sequences = ['Positive', 'Negative', 'Zero']
        magnitudes = [
            sym_comp_data['positive_sequence']['magnitude'],
            sym_comp_data['negative_sequence']['magnitude'],
            sym_comp_data['zero_sequence']['magnitude']
        ]
        angles = [
            sym_comp_data['positive_sequence']['angle'],
            sym_comp_data['negative_sequence']['angle'],
            sym_comp_data['zero_sequence']['angle']
        ]
    
        # 1. Bar chart of sequence magnitudes
        fig.add_trace(
            go.Bar(
                x=sequences,
                y=magnitudes,
                text=[f"{m:.4f}" for m in magnitudes],
                textposition='auto',
                marker_color=['#2ca02c', '#d62728', '#1f77b4']
            ),
            row=1, col=1
        )
    
        # 2. Polar plot for phasor representation
        colors = ['#2ca02c', '#d62728', '#1f77b4']
        for i, seq in enumerate(sequences):
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, magnitudes[i]],
                    theta=[0, angles[i]],
                    mode='lines+markers',
                    line=dict(width=3, color=colors[i]),
                    marker=dict(size=[8, 12], color=colors[i]),
                    name=f"{seq} Sequence"
                ),
                row=1, col=2
            )
    
        # Update layout
        fig.update_layout(
            title=f"Symmetrical Components Analysis<br>Negative Sequence Unbalance: {sym_comp_data['unbalance_factors']['negative_sequence']:.2f}%, Zero Sequence Unbalance: {sym_comp_data['unbalance_factors']['zero_sequence']:.2f}%",
            height=500,
            template='plotly_dark' if dark_mode else 'plotly_white'
        )
    
        # Update axes
        fig.update_xaxes(title_text="Sequence Component", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude", row=1, col=1)
    
        # Update polar axis
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(magnitudes) * 1.1]
                ),
                angularaxis=dict(
                    visible=True,
                    direction="clockwise"
                )
            )
        )
    
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
    
        return fig
    
    def create_coherence_plot(self, coherence_data, dark_mode=False):
        """Create coherence analysis plot"""
        if coherence_data is None or 'frequency' not in coherence_data:
            return go.Figure()
        
        # Extract data
        f = coherence_data['frequency']
        amp_coherence = coherence_data['amplitude_coherence']
        phase_coherence = coherence_data['phase_coherence']
        band_analysis = coherence_data['band_analysis']
    
        # Create figure with subplots
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Amplitude Coherence', 
                'Phase Coherence',
                'Coherence by Frequency Band',
                'Band Analysis Summary'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "table"}]
            ]
        )
    
        # 1. Amplitude Coherence Plot
        fig.add_trace(
            go.Scatter(
                x=f,
                y=amp_coherence,
                mode='lines',
                name='Amplitude Coherence',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
        # 2. Phase Coherence Plot
        fig.add_trace(
            go.Scatter(
                x=f,
                y=phase_coherence,
                mode='lines',
                name='Phase Coherence',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
    
        # 3. Coherence by Frequency Band
        band_names = [b['band'] for b in band_analysis]
        amp_means = [b['amplitude_coherence']['mean'] for b in band_analysis]
        phase_means = [b['phase_coherence']['mean'] for b in band_analysis]
    
        fig.add_trace(
            go.Bar(
                x=band_names,
                y=amp_means,
                name='Amplitude Coherence',
                marker_color='blue'
            ),
            row=2, col=1
        )
    
        fig.add_trace(
            go.Bar(
                x=band_names,
                y=phase_means,
                name='Phase Coherence',
                marker_color='red'
            ),
            row=2, col=1
        )
    
        # 4. Band Analysis Table
        table_headers = ['Frequency Band', 'Amp Coh Mean', 'Amp Coh Max', 'Phase Coh Mean', 'Phase Coh Max']
        table_data = []
    
        for band in band_analysis:
            table_data.append([
                band['band'],
                f"{band['amplitude_coherence']['mean']:.3f}",
                f"{band['amplitude_coherence']['max']:.3f}",
                f"{band['phase_coherence']['mean']:.3f}",
                f"{band['phase_coherence']['max']:.3f}"
            ])
    
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    font=dict(size=12, color='white' if dark_mode else 'black'),
                    fill_color='rgba(0, 102, 204, 0.8)'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    font=dict(size=11),
                    fill_color='rgba(242, 242, 242, 1)' if not dark_mode else 'rgba(50, 50, 50, 1)'
                )
            ),
            row=2, col=2
        )
    
        # Update layout
        fig.update_layout(
            title='Coherence Analysis',
            barmode='group',
            template='plotly_dark' if dark_mode else 'plotly_white',
            height=800
        )
    
        # Update x and y-axis titles
        fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=1)
        fig.update_yaxes(title_text='Coherence', row=1, col=1)
    
        fig.update_xaxes(title_text='Frequency (Hz)', row=1, col=2)
        fig.update_yaxes(title_text='Coherence', row=1, col=2)
    
        fig.update_xaxes(title_text='Frequency Band', row=2, col=1)
        fig.update_yaxes(title_text='Mean Coherence', row=2, col=1)
    
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
    
        return fig
    
    def create_multi_phase_fft_plot(self, multi_phase_data, plot_style='overlay', dark_mode=False, phase_map=None):
        """Create multi-phase FFT comparison plot"""
        if multi_phase_data is None or 'type' not in multi_phase_data or multi_phase_data['type'] != 'fft':
            return go.Figure()
            
        # Extract data
        phase_results = multi_phase_data['phase_results']
        phases = list(phase_results.keys())
        
        if plot_style == 'overlay':
            # Create figure
            fig = go.Figure()
            
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
                
                # Add FFT trace
                fig.add_trace(go.Scatter(
                    x=phase_data['freq'],
                    y=phase_data['magnitude'],
                    mode='lines',
                    name=f'Phase {phase}',
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase FFT Analysis - GB200_2",
                xaxis_title='Frequency (Hz)',
                yaxis_title='Magnitude',
                xaxis_range=[0, 500],  # Limit frequency range
                template='plotly_dark' if dark_mode else 'plotly_white',
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode='closest'
            )
        else:  # Separate
            # Create figure with subplots
            fig = sp.make_subplots(
                rows=len(phases), 
                cols=1,
                subplot_titles=[f'Phase {phase}' for phase in phases],
                vertical_spacing=0.1
            )
            
            # Add traces for each phase
            colors = px.colors.qualitative.Plotly
            for i, phase in enumerate(phases):
                phase_data = phase_results[phase]
                
                # Add FFT trace
                fig.add_trace(
                    go.Scatter(
                        x=phase_data['freq'],
                        y=phase_data['magnitude'],
                        mode='lines',
                        name=f'Phase {phase}',
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=i+1, col=1
                )
                
                # Set x-axis range
                fig.update_xaxes(range=[0, 500], row=i+1, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"Multi-Phase FFT Analysis - GB200_2",
                template='plotly_dark' if dark_mode else 'plotly_white',
                height=300*len(phases),
                showlegend=True,
                hovermode='closest'
            )
            
            # Update y-axis title
            for i in range(len(phases)):
                fig.update_yaxes(title_text='Magnitude', row=i+1, col=1)
            
            # Update x-axis (only show on bottom plot)
            for i in range(len(phases)-1):
                fig.update_xaxes(showticklabels=False, row=i+1, col=1)
            fig.update_xaxes(title_text='Frequency (Hz)', row=len(phases), col=1)
        
        # Apply theme
        fig = self._apply_theme(fig, dark_mode)
        
        return fig
    
    def _apply_theme(self, fig, dark_mode=False):
        """Apply theme colors to plotly figure with proper contrast"""
        if dark_mode:
            fig.update_layout(
                paper_bgcolor=DARK_MODE_STYLES['plot_paper_bg'],
                plot_bgcolor=DARK_MODE_STYLES['plot_bg'],
                font={'color': DARK_MODE_STYLES['text']},
                title={'font': {'color': DARK_MODE_STYLES['text']}},
                legend={
                    'font': {'color': DARK_MODE_STYLES['text']},
                    'bgcolor': DARK_MODE_STYLES['plot_bg'],
                    'bordercolor': DARK_MODE_STYLES['plot_grid']
                },
                modebar={'bgcolor': 'rgba(45, 45, 45, 0.7)', 'color': DARK_MODE_STYLES['text']},
                hoverlabel={'bgcolor': DARK_MODE_STYLES['tooltip_bg'], 'font': {'color': DARK_MODE_STYLES['tooltip_text']}},
                coloraxis={'colorbar': {'tickfont': {'color': DARK_MODE_STYLES['text']}}}
            )
        
            # Update all xaxes
            fig.update_xaxes(
                gridcolor=DARK_MODE_STYLES['plot_grid'],
                zerolinecolor=DARK_MODE_STYLES['plot_grid'],
                linecolor=DARK_MODE_STYLES['plot_grid'],
                title={'font': {'color': DARK_MODE_STYLES['text']}},
                tickfont={'color': DARK_MODE_STYLES['text']}
            )
        
            # Update all yaxes
            fig.update_yaxes(
                gridcolor=DARK_MODE_STYLES['plot_grid'],
                zerolinecolor=DARK_MODE_STYLES['plot_grid'],
                linecolor=DARK_MODE_STYLES['plot_grid'],
                title={'font': {'color': DARK_MODE_STYLES['text']}},
                tickfont={'color': DARK_MODE_STYLES['text']}
            )
        
            # Specific handling for 3D plots
            if hasattr(fig, 'layout') and hasattr(fig.layout, 'scene'):
                fig.update_layout(
                    scene={
                        'xaxis': {
                            'gridcolor': DARK_MODE_STYLES['plot_grid'],
                            'zerolinecolor': DARK_MODE_STYLES['plot_grid'],
                            'backgroundcolor': DARK_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': DARK_MODE_STYLES['text']}},
                            'tickfont': {'color': DARK_MODE_STYLES['text']}
                        },
                        'yaxis': {
                            'gridcolor': DARK_MODE_STYLES['plot_grid'],
                            'zerolinecolor': DARK_MODE_STYLES['plot_grid'],
                            'backgroundcolor': DARK_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': DARK_MODE_STYLES['text']}},
                            'tickfont': {'color': DARK_MODE_STYLES['text']}
                        },
                        'zaxis': {
                            'gridcolor': DARK_MODE_STYLES['plot_grid'],
                            'zerolinecolor': DARK_MODE_STYLES['plot_grid'],
                            'backgroundcolor': DARK_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': DARK_MODE_STYLES['text']}},
                            'tickfont': {'color': DARK_MODE_STYLES['text']}
                        }
                    }
                )
        else:
            fig.update_layout(
                paper_bgcolor=LIGHT_MODE_STYLES['plot_paper_bg'],
                plot_bgcolor=LIGHT_MODE_STYLES['plot_bg'],
                font={'color': LIGHT_MODE_STYLES['text']},
                title={'font': {'color': LIGHT_MODE_STYLES['text']}},
                legend={
                    'font': {'color': LIGHT_MODE_STYLES['text']},
                    'bgcolor': LIGHT_MODE_STYLES['plot_bg'],
                    'bordercolor': LIGHT_MODE_STYLES['plot_grid']
                },
                modebar={'bgcolor': 'rgba(255, 255, 255, 0.7)', 'color': LIGHT_MODE_STYLES['text']},
                hoverlabel={'bgcolor': LIGHT_MODE_STYLES['tooltip_bg'], 'font': {'color': LIGHT_MODE_STYLES['tooltip_text']}},
                coloraxis={'colorbar': {'tickfont': {'color': LIGHT_MODE_STYLES['text']}}}
            )
        
            # Update all xaxes
            fig.update_xaxes(
                gridcolor=LIGHT_MODE_STYLES['plot_grid'],
                zerolinecolor=LIGHT_MODE_STYLES['plot_grid'],
                linecolor=LIGHT_MODE_STYLES['plot_grid'],
                title={'font': {'color': LIGHT_MODE_STYLES['text']}},
                tickfont={'color': LIGHT_MODE_STYLES['text']}
            )
        
            # Update all yaxes
            fig.update_yaxes(
                gridcolor=LIGHT_MODE_STYLES['plot_grid'],
                zerolinecolor=LIGHT_MODE_STYLES['plot_grid'],
                linecolor=LIGHT_MODE_STYLES['plot_grid'],
                title={'font': {'color': LIGHT_MODE_STYLES['text']}},
                tickfont={'color': LIGHT_MODE_STYLES['text']}
            )
        
            # Specific handling for 3D plots
            if hasattr(fig, 'layout') and hasattr(fig.layout, 'scene'):
                fig.update_layout(
                    scene={
                        'xaxis': {
                            'gridcolor': LIGHT_MODE_STYLES['plot_grid'],
                            'zerolinecolor': LIGHT_MODE_STYLES['plot_grid'],
                            'backgroundcolor': LIGHT_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': LIGHT_MODE_STYLES['text']}},
                            'tickfont': {'color': LIGHT_MODE_STYLES['text']}
                        },
                        'yaxis': {
                            'gridcolor': LIGHT_MODE_STYLES['plot_grid'],
                            'zerolinecolor': LIGHT_MODE_STYLES['plot_grid'],
                            'backgroundcolor': LIGHT_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': LIGHT_MODE_STYLES['text']}},
                            'tickfont': {'color': LIGHT_MODE_STYLES['text']}
                        },
                        'zaxis': {
                            'gridcolor': LIGHT_MODE_STYLES['plot_grid'],
                            'zerolinecolor': LIGHT_MODE_STYLES['plot_grid'],
                            'backgroundcolor': LIGHT_MODE_STYLES['plot_bg'],
                            'showbackground': True,
                            'title': {'font': {'color': LIGHT_MODE_STYLES['text']}},
                            'tickfont': {'color': LIGHT_MODE_STYLES['text']}
                        }
                    }
                )
    
        return fig