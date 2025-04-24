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
    """Dedicated transient analysis"""
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
            
    def compute_waveform_distortion(self, data):
        """Analyze waveform distortion parameters"""
        if data is None or 'current' not in data or len(data['current']) == 0:
            return None
            
        try:
            current = data['current']
            
            # Compute harmonics first
            harmonics_data = self.compute_harmonics(data)
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