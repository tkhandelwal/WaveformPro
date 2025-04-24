import numpy as np
from scipy import signal
import pywt  # For wavelet transform
import logging
import traceback

logger = logging.getLogger(__name__)

class SignalProcessor:
    def calculate_metrics(self, data):
        """Calculate standard metrics for a signal"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        current = data['current']
        try:
            # Basic metrics
            metrics = {
                'mean': np.mean(current),
                'median': np.median(current),
                'std': np.std(current),
                'rms': np.sqrt(np.mean(np.square(current))),
                'peak_to_peak': np.max(current) - np.min(current),
                'crest_factor': np.max(np.abs(current)) / (np.sqrt(np.mean(np.square(current))) + 1e-10),
                'form_factor': np.sqrt(np.mean(np.square(current))) / (np.mean(np.abs(current)) + 1e-10),
                'max_current': np.max(current),
                'min_current': np.min(current),
                'num_samples': len(current),
                'duration': data['time'][-1] - data['time'][0] if len(data['time']) > 0 else 0
            }
            
            # Add frequency domain metrics if sample rate is available
            if 'sample_rate' in data and data['sample_rate'] > 0:
                sample_rate = data['sample_rate']
                # Estimate dominant frequency
                freqs, psd = signal.welch(current, fs=sample_rate, nperseg=min(1024, len(current)))
                if len(psd) > 0:
                    idx_max = np.argmax(psd)
                    metrics['dominant_freq'] = freqs[idx_max]
                    metrics['dominant_power'] = psd[idx_max]
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return None
    
    def compute_fft(self, data, window_type='hann'):
        """Compute FFT with proper windowing and scaling"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            n = len(data['current'])
            sample_interval = data['sample_interval']
            
            # Apply window to reduce spectral leakage
            if window_type == 'hann':
                window = np.hanning(n)
            elif window_type == 'hamming':
                window = np.hamming(n)
            elif window_type == 'blackman':
                window = np.blackman(n)
            elif window_type == 'flattop':
                window = signal.flattop(n)
            else:
                window = np.ones(n)  # Rectangular window (no window)
                
            windowed_data = data['current'] * window
            
            # Perform FFT
            fft_result = np.fft.rfft(windowed_data)
            fft_freq = np.fft.rfftfreq(n, d=sample_interval)
            
            # Magnitude
            fft_magnitude = np.abs(fft_result) * 2.0 / n
            
            # Compute dB scale (with floor to avoid log(0))
            fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)
            
            return {
                'freq': fft_freq,
                'magnitude': fft_magnitude,
                'magnitude_db': fft_magnitude_db,
                'phase': np.angle(fft_result),
                'window_type': window_type,
                'n_samples': n,
                'sample_interval': sample_interval
            }
        except Exception as e:
            logger.error(f"Error computing FFT: {e}")
            traceback.print_exc()
            return None
    
    def compute_spectrogram(self, data, nperseg=256, noverlap=None):
        """Compute spectrogram for time-frequency analysis"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            # Use scipy's spectrogram function
            sample_rate = 1 / data['sample_interval']
            if noverlap is None:
                noverlap = nperseg // 2
                
            nperseg = min(nperseg, len(data['current']) // 2)
            
            f, t, Sxx = signal.spectrogram(
                data['current'], 
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            
            # Convert to dB scale
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            return {
                'freq': f,
                'time': t,
                'power': Sxx_db,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'sample_rate': sample_rate
            }
        except Exception as e:
            logger.error(f"Error computing spectrogram: {e}")
            traceback.print_exc()
            return None
    
    def compute_stft(self, data, nperseg=256, noverlap=None, window='hann'):
        """Compute Short-Time Fourier Transform for time-frequency analysis"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            # Use scipy's STFT function
            sample_rate = 1 / data['sample_interval']
            if noverlap is None:
                noverlap = nperseg // 2
                
            nperseg = min(nperseg, len(data['current']) // 2)
            
            f, t, Zxx = signal.stft(
                data['current'], 
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window=window
            )
            
            # Magnitude of STFT
            Zxx_mag = np.abs(Zxx)
            
            # Convert to dB scale
            Zxx_db = 20 * np.log10(Zxx_mag + 1e-10)
            
            # Phase of STFT
            Zxx_phase = np.angle(Zxx)
            
            return {
                'freq': f,
                'time': t,
                'magnitude': Zxx_mag,
                'magnitude_db': Zxx_db,
                'phase': Zxx_phase,
                'nperseg': nperseg,
                'noverlap': noverlap,
                'window': window,
                'sample_rate': sample_rate
            }
        except Exception as e:
            logger.error(f"Error computing STFT: {e}")
            traceback.print_exc()
            return None
    
    def compute_envelope(self, data):
        """Compute signal envelope using Hilbert transform"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            # Compute analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(data['current'])
            
            # Envelope is the magnitude of analytic signal
            envelope = np.abs(analytic_signal)
            
            return {
                'time': data['time'],
                'envelope': envelope
            }
        except Exception as e:
            logger.error(f"Error computing envelope: {e}")
            traceback.print_exc()
            return None
    
    def compute_harmonics(self, data, fundamental_freq=60, num_harmonics=40):
        """Compute harmonic spectrum and THD"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            sample_rate = 1 / data['sample_interval']
            
            # Get FFT results
            fft_data = self.compute_fft(data, window_type='hann')
            if fft_data is None:
                return None
                
            freqs = fft_data['freq']
            magnitude = fft_data['magnitude']
            
            # Find fundamental frequency peak
            fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
            fund_magnitude = magnitude[fund_idx]
            
            # Extract harmonics
            harmonics = []
            for h in range(1, num_harmonics + 1):
                harmonic_freq = h * fundamental_freq
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_magnitude = magnitude[idx]
                
                harmonics.append({
                    'harmonic': h,
                    'frequency': freqs[idx],
                    'magnitude': harmonic_magnitude,
                    'phase': fft_data['phase'][idx]
                })
            
            # Calculate THD
            sum_squares = 0
            for h in harmonics[1:]:  # Skip fundamental
                sum_squares += h['magnitude'] ** 2
                
            thd = np.sqrt(sum_squares) / (fund_magnitude + 1e-10) * 100
            
            return {
                'harmonics': harmonics,
                'thd': thd,
                'fundamental_freq': fundamental_freq,
                'fundamental_magnitude': fund_magnitude
            }
        except Exception as e:
            logger.error(f"Error computing harmonics: {e}")
            traceback.print_exc()
            return None
            
    def compute_interharmonics(self, data, fundamental_freq=60, num_groups=10):
        """Compute interharmonics analysis between harmonic frequencies"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            # Get FFT results with high resolution
            fft_data = self.compute_fft(data, window_type='blackman')
            if fft_data is None:
                return None
                
            freqs = fft_data['freq']
            magnitude = fft_data['magnitude']
            
            # Extract interharmonic groups
            interharmonic_groups = []
            
            for i in range(1, num_groups + 1):
                # Define frequency range between harmonics
                f_start = i * fundamental_freq
                f_end = (i + 1) * fundamental_freq
                
                # Find indices in this range
                indices = np.where((freqs > f_start) & (freqs < f_end))[0]
                
                if len(indices) > 0:
                    # Find the maximum interharmonic in this range
                    max_idx = indices[np.argmax(magnitude[indices])]
                    
                    interharmonic_groups.append({
                        'group': i,
                        'frequency': freqs[max_idx],
                        'magnitude': magnitude[max_idx],
                        'phase': fft_data['phase'][max_idx]
                    })
            
            # Calculate total interharmonic distortion (similar to THD but for interharmonics)
            fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
            fund_magnitude = magnitude[fund_idx]
            
            sum_squares_ih = 0
            for ih in interharmonic_groups:
                sum_squares_ih += ih['magnitude'] ** 2
                
            tid = np.sqrt(sum_squares_ih) / (fund_magnitude + 1e-10) * 100
            
            return {
                'interharmonic_groups': interharmonic_groups,
                'tid': tid,  # Total interharmonic distortion
                'fundamental_freq': fundamental_freq,
                'fundamental_magnitude': fund_magnitude
            }
        except Exception as e:
            logger.error(f"Error computing interharmonics: {e}")
            traceback.print_exc()
            return None
    
    def compute_wavelet(self, data, wavelet='db4', level=5):
        """Compute wavelet decomposition"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            # Get the wavelet coefficients
            coeffs = pywt.wavedec(data['current'], wavelet, level=level)
            
            # Reconstruct signals at each level
            reconstructed = []
            for i in range(level + 1):
                coeff_list = [None] * (level + 1)
                coeff_list[i] = coeffs[i]
                
                # Replace all other coefficients with zeros
                for j in range(level + 1):
                    if j != i:
                        if j == 0:  # Approximation
                            coeff_list[j] = np.zeros_like(coeffs[j])
                        else:  # Details
                            coeff_list[j] = np.zeros_like(coeffs[j])
                
                # Reconstruct signal
                reconstructed.append(pywt.waverec(coeff_list, wavelet))
            
            # Ensure all reconstructed signals have the same length
            min_length = min(len(sig) for sig in reconstructed)
            reconstructed = [sig[:min_length] for sig in reconstructed]
            
            # Create time array for reconstructed signals
            if len(data['time']) >= min_length:
                time = data['time'][:min_length]
            else:
                # Interpolate if necessary
                time = np.linspace(data['time'][0], data['time'][-1], min_length)
            
            # Prepare result
            result = {
                'time': time,
                'wavelet_type': wavelet,
                'level': level,
                'approximation': reconstructed[0],
                'details': reconstructed[1:],
                'coefficients': coeffs
            }
            
            return result
        except Exception as e:
            logger.error(f"Error computing wavelet decomposition: {e}")
            traceback.print_exc()
            return None
            
    def analyze_power_quality(self, data, analysis_type='flicker', sensitivity=2):
        """Analyze power quality issues (flicker, transients, etc.)"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            current = data['current']
            time = data['time']
            sample_rate = 1 / data['sample_interval']
            
            if analysis_type == 'flicker':
                # Simple flicker analysis using envelope
                envelope_data = self.compute_envelope(data)
                if envelope_data is None:
                    return None
                    
                envelope = envelope_data['envelope']
                
                # Calculate flicker metrics
                mean_envelope = np.mean(envelope)
                std_envelope = np.std(envelope)
                
                # Short-term flicker severity (Pst) - simplified approximation
                # In reality, Pst calculation is more complex and follows IEC 61000-4-15
                envelope_normalized = envelope / mean_envelope
                Pst = np.percentile(envelope_normalized, 95) - np.percentile(envelope_normalized, 5)
                
                # Long-term flicker severity (Plt) - simplified approximation
                # Typically Plt is calculated over 2 hours using 12 Pst values
                Plt = 0.8 * Pst  # Simplified approximation
                
                # Maximum and average deviation
                max_deviation = np.max(np.abs(envelope - mean_envelope))
                avg_deviation = np.mean(np.abs(envelope - mean_envelope))
                
                return {
                    'type': 'flicker',
                    'Pst': Pst,
                    'Plt': Plt,
                    'max_deviation': max_deviation,
                    'avg_deviation': avg_deviation,
                    'time': time,
                    'envelope': envelope,
                    'current': current
                }
                
            return None
        except Exception as e:
            logger.error(f"Error analyzing power quality: {e}")
            traceback.print_exc()
            return None
    
    def analyze_transients(self, data, sensitivity=2, window_size=20):
        """Analyze transients in the signal"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
        
        try:
            current = data['current']
            time = data['time']
            sample_rate = 1 / data['sample_interval']
        
            # Calculate RMS value for baseline
            rms_value = np.sqrt(np.mean(np.square(current)))
        
            # Dynamic threshold based on sensitivity
            threshold = sensitivity * rms_value
        
            # Detect transients using threshold detection
            # Find peaks exceeding threshold
            peaks, _ = signal.find_peaks(np.abs(current), height=threshold)
        
            # Group peaks into events (peaks within window_size samples of each other)
            events = []
            if len(peaks) > 0:
                event_start_idx = peaks[0]
                event_peak_idx = peaks[0]
                event_peak_value = current[peaks[0]]
            
                for i in range(1, len(peaks)):
                    # If next peak is far enough, consider it a new event
                    if (peaks[i] - peaks[i-1]) > window_size:
                        # Record the previous event
                        event_end_idx = peaks[i-1]
                        duration_ms = (time[event_end_idx] - time[event_start_idx]) * 1000  # convert to ms
                    
                        # Calculate transient energy
                        event_indices = range(event_start_idx, event_end_idx + 1)
                        energy = np.sum(np.square(current[event_indices]))
                    
                        events.append({
                            'start_time': time[event_start_idx],
                            'end_time': time[event_end_idx],
                            'duration': duration_ms,
                            'peak_value': event_peak_value,
                            'peak_time': time[event_peak_idx],
                            'energy': energy,
                            'class': self._classify_transient(duration_ms, event_peak_value, rms_value)
                        })
                    
                        # Start a new event
                        event_start_idx = peaks[i]
                        event_peak_idx = peaks[i]
                        event_peak_value = current[peaks[i]]
                    else:
                        # Update peak value if higher
                        if abs(current[peaks[i]]) > abs(event_peak_value):
                            event_peak_idx = peaks[i]
                            event_peak_value = current[peaks[i]]
            
                # Add the last event
                if len(peaks) > 0:
                    event_end_idx = peaks[-1]
                    duration_ms = (time[event_end_idx] - time[event_start_idx]) * 1000  # convert to ms
                
                    # Calculate transient energy
                    event_indices = range(event_start_idx, event_end_idx + 1)
                    energy = np.sum(np.square(current[event_indices]))
                
                    events.append({
                        'start_time': time[event_start_idx],
                        'end_time': time[event_end_idx],
                        'duration': duration_ms,
                        'peak_value': event_peak_value,
                        'peak_time': time[event_peak_idx],
                        'energy': energy,
                        'class': self._classify_transient(duration_ms, event_peak_value, rms_value)
                    })
        
            # Calculate statistics
            total_energy = sum(event['energy'] for event in events) if events else 0
            max_peak = max([abs(event['peak_value']) for event in events]) if events else 0
            avg_duration = np.mean([event['duration'] for event in events]) if events else 0
        
            return {
                'type': 'transients',
                'events': events,
                'threshold': threshold,
                'detected_count': len(events),
                'time': time,
                'current': current,
                'baseline_rms': rms_value,
                'total_energy': total_energy,
                'max_peak': max_peak,
                'avg_duration': avg_duration
            }
        except Exception as e:
            logger.error(f"Error analyzing transients: {e}")
            traceback.print_exc()
            return None

    def _classify_transient(self, duration, peak_value, rms_value):
        """Classify transient based on duration and amplitude"""
        # Simple classification based on duration
        if duration < 1.0:  # less than 1ms
            return 'Impulse'
        elif duration < 10.0:  # less than 10ms
            return 'Oscillatory'
        else:
            # Distinguish between swell and dip based on polarity
            if peak_value > 0:
                return 'Swell'
            else:
                return 'Dip'
            
    
    def compute_cepstrum(self, data):
        """Compute cepstrum analysis for detecting periodicities"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
        
        try:
            # Get the magnitude spectrum
            fft_data = self.compute_fft(data, window_type='hann')
            if fft_data is None:
                return None
        
            # Compute the real cepstrum
            # Take log of magnitude spectrum
            log_mag = np.log(fft_data['magnitude'] + 1e-10)
        
            # Take inverse FFT of log magnitude spectrum
            cepstrum = np.fft.irfft(log_mag)
        
            # Create quefrency axis (in seconds)
            n = len(cepstrum)
            quefrency = np.arange(n) * data['sample_interval']
        
            # Find peaks in cepstrum
            peaks, properties = signal.find_peaks(cepstrum, height=0.1*np.max(cepstrum))
        
            peak_data = []
            for i, peak_idx in enumerate(peaks):
                # Convert quefrency to frequency
                if quefrency[peak_idx] > 0:
                    freq = 1 / quefrency[peak_idx]
                    peak_data.append({
                        'quefrency': quefrency[peak_idx],
                        'frequency': freq,
                        'amplitude': cepstrum[peak_idx]
                    })
        
            # Sort by amplitude
            peak_data = sorted(peak_data, key=lambda x: x['amplitude'], reverse=True)
        
            return {
                'quefrency': quefrency,
                'cepstrum': cepstrum,
                'peaks': peak_data[:10]  # Return top 10 peaks
            }
        except Exception as e:
            logger.error(f"Error computing cepstrum: {e}")
            traceback.print_exc()
            return None
    
    def analyze_multi_phase(self, data_dict, analysis_type='thd', phases=None):
        """Analyze multiple phases together"""
        if not data_dict or not isinstance(data_dict, dict):
            return None
        
        # If phases not specified, use all keys in data_dict
        if phases is None:
            phases = list(data_dict.keys())
    
        # Filter only specified phases
        phase_data = {phase: data_dict[phase] for phase in phases if phase in data_dict}
    
        if not phase_data:
            return None
        
        try:
            if analysis_type == 'thd':
                # Harmonics comparison
                results = {}
            
                for phase, data in phase_data.items():
                    # Get harmonics for this phase
                    harmonic_data = self.compute_harmonics(data)
                
                    if harmonic_data:
                        results[phase] = {
                            'thd': harmonic_data['thd'],
                            'fundamental_freq': harmonic_data['fundamental_freq'],
                            'fundamental_magnitude': harmonic_data['fundamental_magnitude'],
                            'harmonics': harmonic_data['harmonics']
                        }
            
                return {
                    'type': 'thd',
                    'phase_results': results
                }
            
            elif analysis_type == 'fft':
                # Compare FFT for multiple phases
                results = {}
            
                for phase, data in phase_data.items():
                    # Get FFT for this phase
                    fft_data = self.compute_fft(data)
                
                    if fft_data:
                        results[phase] = {
                            'freq': fft_data['freq'],
                            'magnitude': fft_data['magnitude'],
                            'magnitude_db': fft_data['magnitude_db']
                        }
            
                return {
                    'type': 'fft',
                    'phase_results': results
                }
            
            elif analysis_type == 'time':
                # Compare time domain data
                results = {}
            
                for phase, data in phase_data.items():
                    # Get metrics for this phase
                    metrics = self.calculate_metrics(data)
                
                    if metrics:
                        results[phase] = {
                            'metrics': metrics,
                            'time': data['time'],
                            'current': data['current']
                        }
            
                return {
                    'type': 'time',
                    'phase_results': results
                }
            
            elif analysis_type == 'correlation':
                # Compute correlation between phases
                results = {}
                correlation_matrix = {}
            
                # First, collect all waveforms
                waveforms = {}
                for phase, data in phase_data.items():
                    waveforms[phase] = data['current']
            
                # Compute correlation between all pairs
                for phase1 in phases:
                    if phase1 not in correlation_matrix:
                        correlation_matrix[phase1] = {}
                
                    for phase2 in phases:
                        if phase1 in waveforms and phase2 in waveforms:
                            # Compute correlation coefficient
                            corr = np.corrcoef(waveforms[phase1], waveforms[phase2])[0, 1]
                            correlation_matrix[phase1][phase2] = corr
            
                # Compute phase angles from correlation
                phase_angles = {}
                reference_phase = phases[0]
            
                for phase in phases:
                    if phase in waveforms and reference_phase in waveforms:
                        # Cross-correlation to find time delay
                        cross_corr = signal.correlate(waveforms[reference_phase], waveforms[phase])
                        lags = signal.correlation_lags(len(waveforms[reference_phase]), len(waveforms[phase]))
                    
                        # Find the lag with maximum correlation
                        max_corr_idx = np.argmax(cross_corr)
                        lag = lags[max_corr_idx]
                    
                        # Convert lag to phase angle (assuming 60Hz)
                        # For a complete cycle of 60Hz, there should be sample_rate/60 samples
                        sample_rate = 1 / phase_data[phase]['sample_interval']
                        samples_per_cycle = sample_rate / 60
                    
                        # Convert lag to phase angle in degrees
                        phase_angle = (lag / samples_per_cycle) * 360
                    
                        # Normalize to range [-180, 180]
                        if phase_angle > 180:
                            phase_angle -= 360
                        elif phase_angle < -180:
                            phase_angle += 360
                    
                        phase_angles[phase] = phase_angle
            
                # Store results
                results = {
                    'correlation_matrix': correlation_matrix,
                    'phase_angles': phase_angles,
                    'reference_phase': reference_phase
                }
            
                return {
                    'type': 'correlation',
                    'correlation_results': results
                }
            
            elif analysis_type == 'impedance':
                # Compute apparent impedance for each phase
                # This is a simplified version, assuming we have voltage data
                # In reality, we would need both voltage and current measurements
                results = {}
            
                for phase, data in phase_data.items():
                    # Get FFT for this phase
                    fft_data = self.compute_fft(data)
                
                    if fft_data:
                        # For demo purposes, simulate voltage data
                        # In reality, you would use actual voltage measurements
                        voltage_magnitude = 120 * np.ones_like(fft_data['magnitude'])  # 120V nominal
                    
                        # Compute apparent impedance (voltage/current)
                        impedance = np.divide(
                            voltage_magnitude, 
                            fft_data['magnitude'], 
                            out=np.zeros_like(fft_data['magnitude']), 
                            where=fft_data['magnitude']>1e-10
                        )
                    
                        # Store results
                        results[phase] = {
                            'freq': fft_data['freq'],
                            'impedance': impedance,
                            'impedance_db': 20 * np.log10(impedance + 1e-10)
                        }
            
                return {
                    'type': 'impedance',
                    'phase_results': results
                }
        
            return None
        except Exception as e:
            logger.error(f"Error in multi-phase analysis: {e}")
            traceback.print_exc()
            return None
            
    
    def check_standards_compliance(self, analysis_results):
        """Check compliance with industry power quality standards"""
        compliance = {}
    
        # IEEE 519 Standard for harmonic distortion
        if 'thd' in analysis_results:
            thd = analysis_results['thd']
            compliance['IEEE_519'] = {
                'compliant': thd <= 5.0,
                'value': thd,
                'limit': 5.0,
                'description': 'THD limit for sensitive equipment'
            }
    
        # IEC 61000-3-2 for harmonic current emissions
        if 'harmonics' in analysis_results:
            harmonics = analysis_results['harmonics']
            odd_harmonics_compliant = True
            even_harmonics_compliant = True
        
            # Check specific harmonic limits (simplified)
            for h in harmonics:
                h_num = h['harmonic']
                if h_num > 1:  # Skip fundamental
                    if h_num % 2 == 1:  # Odd harmonics
                        if h_num <= 11 and h['magnitude'] > 0.15:
                            odd_harmonics_compliant = False
                    else:  # Even harmonics
                        if h['magnitude'] > 0.10:
                            even_harmonics_compliant = False
        
            compliance['IEC_61000_3_2'] = {
                'compliant': odd_harmonics_compliant and even_harmonics_compliant,
                'odd_harmonics_compliant': odd_harmonics_compliant,
                'even_harmonics_compliant': even_harmonics_compliant,
                'description': 'Limits for harmonic current emissions'
            }
    
        # Add more standards checks
    
        return compliance

    def predict_equipment_impact(self, analysis_results):
        """Predict impact on equipment based on power quality analysis"""
        impacts = []
    
        # Check THD impact on equipment
        if 'thd' in analysis_results:
            thd = analysis_results['thd']
            if thd > 10:
                impacts.append({
                    'equipment': 'Transformers',
                    'issue': 'Increased heating due to high THD',
                    'risk_level': 'High' if thd > 20 else 'Medium',
                    'recommendation': 'Consider harmonic filtering or derating transformers'
                })
            
            if thd > 5:
                impacts.append({
                    'equipment': 'Motors',
                    'issue': 'Reduced efficiency and increased heating',
                    'risk_level': 'Medium' if thd > 15 else 'Low',
                    'recommendation': 'Monitor motor temperature and vibration'
                })
    
        # Check impact of voltage transients
        if 'transients' in analysis_results and analysis_results['detected_count'] > 0:
            transient_count = analysis_results['detected_count']
            impacts.append({
                'equipment': 'Electronic Equipment',
                'issue': 'Potential damage from voltage transients',
                'risk_level': 'High' if transient_count > 10 else 'Medium',
                'recommendation': 'Install surge protectors or voltage regulators'
            })
    
        # Add more rules for other power quality issues
    
        return impacts

    def calculate_health_score(self, data):
        """Calculate a power quality health score from 0-100"""
        # Get various metrics
        metrics = self.calculate_metrics(data)
        harmonics = self.compute_harmonics(data)
    
        # Define weight for each metric
        weights = {
            'thd': 30,  # THD has high importance
            'crest_factor': 20,
            'form_factor': 15,
            'flicker': 20,
            'transients': 15
        }
    
        # Score each component (higher is better)
        scores = {}
    
        # THD score (lower THD is better)
        thd = harmonics['thd']
        scores['thd'] = max(0, 100 - thd * 2)  # Penalize THD over 50%
    
        # Crest factor score (closer to 1.414 for sine wave is better)
        cf = metrics['crest_factor']
        cf_ideal = 1.414  # Ideal for sine wave
        scores['crest_factor'] = max(0, 100 - 30 * abs(cf - cf_ideal))
    
        # Form factor score (closer to 1.11 for sine wave is better)
        ff = metrics['form_factor']
        ff_ideal = 1.11  # Ideal for sine wave
        scores['form_factor'] = max(0, 100 - 30 * abs(ff - ff_ideal))
    
        # Flicker score (detect voltage fluctuations)
        flicker_data = self.analyze_power_quality(data, 'flicker')
        if flicker_data:
            pst = flicker_data['Pst']
            scores['flicker'] = max(0, 100 - pst * 80)  # Pst > 1.25 gives 0 score
        else:
            scores['flicker'] = 100  # Default if can't measure
    
        # Transient score (fewer transients is better)
        transient_data = self.analyze_transients(data)
        if transient_data:
            transient_count = transient_data['detected_count']
            scores['transients'] = max(0, 100 - transient_count * 5)  # Each transient costs 5 points
        else:
            scores['transients'] = 100  # Default if can't measure
    
        # Calculate weighted average
        total_score = 0
        total_weight = 0
        for metric, score in scores.items():
            total_score += score * weights.get(metric, 0)
            total_weight += weights.get(metric, 0)
    
        if total_weight > 0:
            final_score = round(total_score / total_weight)
        else:
            final_score = 0
    
        return {
            'overall_score': final_score,
            'component_scores': scores,
            'interpretation': self._interpret_health_score(final_score)
        }

    def detect_anomalies(self, data, sensitivity=0.8):
        """Detect anomalies in power quality data using machine learning"""
        from sklearn.ensemble import IsolationForest
    
        # Extract features (e.g., RMS, THD, crest factor)
        features = []
        for i in range(0, len(data['current']), 100):  # Sample every 100 points
            segment = data['current'][i:i+100]
            if len(segment) < 20:  # Skip small segments
                continue
            
            # Calculate features for this segment
            rms = np.sqrt(np.mean(np.square(segment)))
            peak = np.max(np.abs(segment))
            crest = peak / rms if rms > 0 else 0
        
            # More features can be added
            features.append([rms, peak, crest])
    
        # Apply Isolation Forest for anomaly detection
        if len(features) > 10:  # Need enough samples
            clf = IsolationForest(contamination=sensitivity)
            outliers = clf.fit_predict(features)
        
            # Find segments with anomalies
            anomalies = []
            j = 0
            for i in range(0, len(data['current']), 100):
                if j >= len(outliers):
                    break
                
                if outliers[j] == -1:  # -1 indicates anomaly
                    anomalies.append({
                        'start': i,
                        'end': min(i+100, len(data['current'])),
                        'time_start': data['time'][i],
                        'time_end': data['time'][min(i+100, len(data['current'])-1)]
                    })
                j += 1
        
            return anomalies
        return []

    def _interpret_health_score(self, score):
        """Interpret the health score"""
        if score >= 90:
            return "Excellent power quality with minimal distortion"
        elif score >= 80:
            return "Good power quality with minor distortion"
        elif score >= 70:
            return "Acceptable power quality with moderate distortion"
        elif score >= 60:
            return "Fair power quality with noticeable distortion"
        elif score >= 50:
            return "Poor power quality with significant distortion"
        else:
            return "Critical power quality issues requiring immediate attention"

    def calculate_power_quality_index(self, data, harmonics_data=None, transient_data=None):
        """Calculate a comprehensive Power Quality Index from multiple metrics"""
        try:
            if data is None:
                return None
            
            metrics = {}
        
            # Basic time domain metrics
            time_metrics = self.calculate_metrics(data)
        
            # Harmonics metrics
            if harmonics_data is None and data is not None:
                harmonics_data = self.compute_harmonics(data)
            
            # Transient metrics
            if transient_data is None and data is not None:
                transient_data = self.analyze_transients(data)
            
            # Calculate individual indices (0-100 scale where 100 is perfect)
        
            # 1. Crest Factor Index (ideal is sqrt(2) ≈ 1.414 for sine wave)
            cf = time_metrics.get('crest_factor', 0)
            cf_ideal = 1.414
            cf_index = max(0, 100 - 30 * abs(cf - cf_ideal))
        
            # 2. Form Factor Index (ideal is 1.11 for sine wave)
            ff = time_metrics.get('form_factor', 0)
            ff_ideal = 1.11
            ff_index = max(0, 100 - 40 * abs(ff - ff_ideal))
        
            # 3. THD Index (ideal is 0%)
            thd = harmonics_data.get('thd', 0) if harmonics_data else 0
            thd_index = max(0, 100 - thd)
        
            # 4. Transient Index (ideal is 0 transients)
            transient_count = transient_data.get('detected_count', 0) if transient_data else 0
            transient_index = max(0, 100 - 10 * transient_count)
        
            # Combine indices with weights
            pqi = (
                0.25 * cf_index + 
                0.20 * ff_index + 
                0.40 * thd_index + 
                0.15 * transient_index
            )
        
            # Classify quality
            quality_level = "Excellent"
            if pqi < 40:
                quality_level = "Poor"
            elif pqi < 60:
                quality_level = "Fair"
            elif pqi < 80:
                quality_level = "Good"
            elif pqi < 90:
                quality_level = "Very Good"
            
            return {
                'pqi': pqi,
                'quality_level': quality_level,
                'components': {
                    'crest_factor_index': cf_index,
                    'form_factor_index': ff_index,
                    'thd_index': thd_index,
                    'transient_index': transient_index
                },
                'raw_metrics': {
                    'crest_factor': cf,
                    'form_factor': ff,
                    'thd': thd,
                    'transient_count': transient_count
                }
            }
        except Exception as e:
            logger.error(f"Error calculating power quality index: {e}")
            return None

    def compute_symmetrical_components(self, phase_data):
        """Compute symmetrical components for three-phase system"""
        try:
            # Check if we have three phases
            if len(phase_data) != 3:
                logger.warning("Symmetrical components require exactly three phases")
                return None
            
            phases = list(phase_data.keys())
        
            # Get FFT for each phase
            fft_results = {}
            for phase, data in phase_data.items():
                fft_results[phase] = self.compute_fft(data)
            
            # Find fundamental frequency components
            fundamental_freq = 60  # Default
            fundamental_indices = {}
            fundamental_vectors = {}
        
            for phase, fft_data in fft_results.items():
                freqs = fft_data['freq']
                magnitudes = fft_data['magnitude']
                phases_rad = fft_data['phase']
            
                # Find index closest to 60Hz
                idx = np.argmin(np.abs(freqs - fundamental_freq))
                fundamental_indices[phase] = idx
            
                # Get complex vector (magnitude and phase)
                magnitude = magnitudes[idx]
                phase_angle = phases_rad[idx]
            
                # Create complex representation
                fundamental_vectors[phase] = magnitude * np.exp(1j * phase_angle)
            
            # Create Fortescue transformation
            a = np.exp(1j * 2 * np.pi / 3)  # 120° operator
        
            # Get the complex vectors in order
            Va = fundamental_vectors[phases[0]]
            Vb = fundamental_vectors[phases[1]]
            Vc = fundamental_vectors[phases[2]]
        
            # Compute symmetrical components
            V0 = (Va + Vb + Vc) / 3  # Zero sequence
            V1 = (Va + a * Vb + a**2 * Vc) / 3  # Positive sequence
            V2 = (Va + a**2 * Vb + a * Vc) / 3  # Negative sequence
        
            # Calculate magnitudes and angles
            V0_mag = np.abs(V0)
            V1_mag = np.abs(V1)
            V2_mag = np.abs(V2)
        
            V0_ang = np.angle(V0, deg=True)
            V1_ang = np.angle(V1, deg=True)
            V2_ang = np.angle(V2, deg=True)
        
            # Calculate unbalance factors
            if V1_mag > 0:
                negative_sequence_unbalance = (V2_mag / V1_mag) * 100
                zero_sequence_unbalance = (V0_mag / V1_mag) * 100
            else:
                negative_sequence_unbalance = 0
                zero_sequence_unbalance = 0
            
            return {
                'zero_sequence': {
                    'magnitude': V0_mag,
                    'angle': V0_ang,
                    'complex': V0
                },
                'positive_sequence': {
                    'magnitude': V1_mag,
                    'angle': V1_ang,
                    'complex': V1
                },
                'negative_sequence': {
                    'magnitude': V2_mag,
                    'angle': V2_ang,
                    'complex': V2
                },
                'unbalance_factors': {
                    'negative_sequence': negative_sequence_unbalance,
                    'zero_sequence': zero_sequence_unbalance
                }
            }
        except Exception as e:
            logger.error(f"Error computing symmetrical components: {e}")
            return None


    def compute_coherence(self, data, segment_length=1024, overlap=512):
        """Compute coherence between different frequency components"""
        try:
            if data is None or 'current' not in data:
                return None
            
            # Create analytic signal using Hilbert transform
            analytic_signal = signal.hilbert(data['current'])
        
            # Extract instantaneous amplitude and phase
            amplitude = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
        
            # Get sample rate
            fs = 1 / data['sample_interval']
        
            # Compute coherence between original signal and amplitude envelope
            f, Cxy = signal.coherence(data['current'], amplitude, fs=fs, nperseg=segment_length, noverlap=overlap)
        
            # Compute coherence between original signal and phase
            f, Cxy_phase = signal.coherence(data['current'], phase, fs=fs, nperseg=segment_length, noverlap=overlap)
        
            # Create frequency bands for analysis
            bands = [
                {'name': 'Fundamental', 'min': 50, 'max': 70},
                {'name': 'Low Harmonics', 'min': 100, 'max': 300},
                {'name': 'Mid Harmonics', 'min': 300, 'max': 600},
                {'name': 'High Harmonics', 'min': 600, 'max': 1200}
            ]
        
            # Analyze coherence in each band
            band_analysis = []
            for band in bands:
                # Find indices in this frequency range
                indices = np.where((f >= band['min']) & (f <= band['max']))[0]
            
                if len(indices) > 0:
                    mean_coh = np.mean(Cxy[indices])
                    max_coh = np.max(Cxy[indices])
                    max_coh_freq = f[indices[np.argmax(Cxy[indices])]]
                
                    mean_coh_phase = np.mean(Cxy_phase[indices])
                    max_coh_phase = np.max(Cxy_phase[indices])
                    max_coh_phase_freq = f[indices[np.argmax(Cxy_phase[indices])]]
                
                    band_analysis.append({
                        'band': band['name'],
                        'frequency_range': [band['min'], band['max']],
                        'amplitude_coherence': {
                            'mean': mean_coh,
                            'max': max_coh,
                            'max_freq': max_coh_freq
                        },
                        'phase_coherence': {
                            'mean': mean_coh_phase,
                            'max': max_coh_phase,
                            'max_freq': max_coh_phase_freq
                        }
                    })
        
            return {
                'frequency': f,
                'amplitude_coherence': Cxy,
                'phase_coherence': Cxy_phase,
                'band_analysis': band_analysis
            }
        except Exception as e:
            logger.error(f"Error computing coherence: {e}")
            return None

    def compute_waveform_distortion(self, data, fundamental_freq=60):
        """Analyze waveform distortion parameters"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
    
        try:
            current = data['current']
    
            # Compute harmonics first
            harmonics_data = self.compute_harmonics(data, fundamental_freq=fundamental_freq)
            if harmonics_data is None:
                return None
    
            # Calculate key distortion parameters
    
            # Total Harmonic Distortion (already in harmonics_data)
            thd = harmonics_data['thd']
    
            # Calculate form factor
            rms_value = np.sqrt(np.mean(np.square(current)))
            mean_abs = np.mean(np.abs(current))
            form_factor = rms_value / mean_abs if mean_abs > 0 else 0
    
            # Calculate crest factor
            peak_value = np.max(np.abs(current))
            crest_factor = peak_value / rms_value if rms_value > 0 else 0
    
            # Calculate K-factor (weighs harmonics by their square of harmonic order)
            harmonics = harmonics_data['harmonics']
            sum_h_squared = 0
            sum_h = 0
    
            for h in harmonics:
                h_order = h['harmonic']
                h_mag_squared = h['magnitude']**2
                sum_h_squared += h_order**2 * h_mag_squared
                sum_h += h_mag_squared
    
            k_factor = sum_h_squared / sum_h if sum_h > 0 else 0
    
            # Calculate transformer derating factor (TDF) - simplified
            tdf = 1 / np.sqrt(1 + (thd/100)**2)
    
            # Group harmonics by type
            even_harmonics = [h for h in harmonics if h['harmonic'] % 2 == 0]
            odd_harmonics = [h for h in harmonics if h['harmonic'] % 2 == 1 and h['harmonic'] > 1]
            triplen_harmonics = [h for h in harmonics if h['harmonic'] % 3 == 0]
    
            even_thd = np.sqrt(sum(h['magnitude']**2 for h in even_harmonics)) / harmonics[0]['magnitude'] * 100
            odd_thd = np.sqrt(sum(h['magnitude']**2 for h in odd_harmonics)) / harmonics[0]['magnitude'] * 100
            triplen_thd = np.sqrt(sum(h['magnitude']**2 for h in triplen_harmonics)) / harmonics[0]['magnitude'] * 100
    
            return {
                'thd': thd,
                'form_factor': form_factor,
                'crest_factor': crest_factor,
                'k_factor': k_factor,
                'transformer_derating_factor': tdf,
                'even_thd': even_thd,
                'odd_thd': odd_thd,
                'triplen_thd': triplen_thd,
                'fundamental': {
                    'frequency': harmonics[0]['frequency'],
                    'magnitude': harmonics[0]['magnitude']
                },
                'harmonics': harmonics
            }
        except Exception as e:
            logger.error(f"Error computing waveform distortion: {e}")
            traceback.print_exc()
            return None