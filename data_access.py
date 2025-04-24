import sqlite3
import numpy as np
import pandas as pd
from functools import lru_cache
import logging
import os
from config import DB_PATH

logger = logging.getLogger(__name__)

class DataAccessLayer:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize database if it doesn't exist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS systems (
                id INTEGER PRIMARY KEY,
                system_type TEXT,
                date TEXT,
                time TEXT
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY,
                gpu_system_id INTEGER,
                phase INTEGER,
                sample_interval REAL,
                vertical_units TEXT,
                record_length INTEGER,
                FOREIGN KEY (gpu_system_id) REFERENCES systems (id)
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_points (
                id INTEGER PRIMARY KEY,
                measurement_id INTEGER,
                time_value REAL,
                current_value REAL,
                FOREIGN KEY (measurement_id) REFERENCES measurements (id)
            )''')
            
            # If the database is empty, insert some demo data
            cursor.execute("SELECT COUNT(*) FROM systems")
            count = cursor.fetchone()[0]
            
            if count == 0:
                self._insert_demo_data(cursor)
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _insert_demo_data(self, cursor):
        """Insert some demo data for testing"""
        # Insert systems
        cursor.execute('''
        INSERT INTO systems (system_type, date, time) VALUES 
            ('NVL36', '2025-01-15', '09:30:00'),
            ('GB200_2', '2025-02-20', '14:45:00')
        ''')
        
        # Get system IDs
        cursor.execute("SELECT id FROM systems WHERE system_type='NVL36'")
        nvl36_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT id FROM systems WHERE system_type='GB200_2'")
        gb200_id = cursor.fetchone()[0]
        
        # Insert measurements
        cursor.execute('''
        INSERT INTO measurements (gpu_system_id, phase, sample_interval, vertical_units, record_length) VALUES
            (?, 1, 0.0001, 'A', 5000),
            (?, 2, 0.0001, 'A', 5000),
            (?, 3, 0.0001, 'A', 5000),
            (?, 1, 0.0001, 'A', 5000),
            (?, 2, 0.0001, 'A', 5000),
            (?, 3, 0.0001, 'A', 5000)
        ''', (nvl36_id, nvl36_id, nvl36_id, gb200_id, gb200_id, gb200_id))
        
        # Get measurement IDs
        measurements = {}
        for system_id, phase in [(nvl36_id, 1), (nvl36_id, 2), (nvl36_id, 3), 
                                 (gb200_id, 1), (gb200_id, 2), (gb200_id, 3)]:
            cursor.execute("SELECT id FROM measurements WHERE gpu_system_id=? AND phase=?", (system_id, phase))
            measurements[(system_id, phase)] = cursor.fetchone()[0]
        
        # Insert some synthetic data points for each measurement
        # (in a real application, you would insert actual measurements)
        for (system_id, phase), measurement_id in measurements.items():
            # Create some synthetic data with phase shift and different harmonics
            time_values = np.linspace(0, 0.5, 5000)  # 0.5 seconds
            
            # Base frequency (60Hz)
            fundamental_freq = 60.0
            
            if system_id == nvl36_id:
                # Create more complex waveform for NVL36
                phase_shift = (phase - 1) * 2 * np.pi / 3  # 120 degrees between phases
                
                if phase == 1:
                    # Phase 1: High 3rd harmonic distortion
                    current_values = 10 * np.sin(2 * np.pi * fundamental_freq * time_values + phase_shift) + \
                                    3 * np.sin(2 * np.pi * 3 * fundamental_freq * time_values + phase_shift) + \
                                    1 * np.sin(2 * np.pi * 5 * fundamental_freq * time_values + phase_shift) + \
                                    0.5 * np.sin(2 * np.pi * 7 * fundamental_freq * time_values + phase_shift)
                elif phase == 2:
                    # Phase 2: High 5th harmonic distortion
                    current_values = 8 * np.sin(2 * np.pi * fundamental_freq * time_values + phase_shift) + \
                                    1 * np.sin(2 * np.pi * 3 * fundamental_freq * time_values + phase_shift) + \
                                    2.5 * np.sin(2 * np.pi * 5 * fundamental_freq * time_values + phase_shift) + \
                                    0.8 * np.sin(2 * np.pi * 7 * fundamental_freq * time_values + phase_shift)
                else:
                    # Phase 3: High 7th harmonic distortion
                    current_values = 9 * np.sin(2 * np.pi * fundamental_freq * time_values + phase_shift) + \
                                    1.5 * np.sin(2 * np.pi * 3 * fundamental_freq * time_values + phase_shift) + \
                                    1 * np.sin(2 * np.pi * 5 * fundamental_freq * time_values + phase_shift) + \
                                    3 * np.sin(2 * np.pi * 7 * fundamental_freq * time_values + phase_shift)
                
                # Add transients to Phase 3
                if phase == 3:
                    for i in range(10):
                        # Add a transient at random positions
                        pos = np.random.randint(100, 4900)
                        width = np.random.randint(5, 20)
                        height = np.random.uniform(15, 20)
                        current_values[pos:pos+width] += height
            else:
                # Create cleaner signal for GB200_2
                phase_shift = (phase - 1) * 2 * np.pi / 3  # 120 degrees between phases
                
                # All phases have similar harmonic content, but different phase shifts
                current_values = 15 * np.sin(2 * np.pi * fundamental_freq * time_values + phase_shift) + \
                                0.3 * np.sin(2 * np.pi * 2 * fundamental_freq * time_values + phase_shift) + \
                                0.7 * np.sin(2 * np.pi * 3 * fundamental_freq * time_values + phase_shift) + \
                                0.2 * np.sin(2 * np.pi * 4 * fundamental_freq * time_values + phase_shift) + \
                                0.6 * np.sin(2 * np.pi * 5 * fundamental_freq * time_values + phase_shift)
                
                # Add some noise
                current_values += np.random.normal(0, 0.2, len(time_values))
            
            # Prepare data for insertion
            data_to_insert = []
            for i in range(len(time_values)):
                data_to_insert.append((measurement_id, time_values[i], current_values[i]))
            
            # Insert in batches to improve performance
            batch_size = 1000
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i+batch_size]
                cursor.executemany(
                    "INSERT INTO data_points (measurement_id, time_value, current_value) VALUES (?, ?, ?)", 
                    batch
                )
        
        logger.info("Demo data inserted successfully")
    
    def _get_connection(self):
        """Establish a connection to the database"""
        return sqlite3.connect(self.db_path)
    
    @lru_cache(maxsize=10)
    def get_system_list(self):
        """Get list of all systems from database with caching for performance"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT system_type FROM systems')
            systems = [row[0] for row in cursor.fetchall()]
            conn.close()
            return systems
        except Exception as e:
            logger.error(f"Error getting system list: {e}")
            return []
    
    @lru_cache(maxsize=20)
    def get_phases_for_system(self, system):
        """Get phases for a specific system with caching for performance"""
        if not system:
            return []
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            SELECT DISTINCT m.phase
            FROM measurements m
            JOIN systems s ON m.gpu_system_id = s.id
            WHERE s.system_type = ?
            ORDER BY m.phase
            ''', (system,))
            phases = [row[0] for row in cursor.fetchall()]
            conn.close()
            return phases
        except Exception as e:
            logger.error(f"Error getting phases for system {system}: {e}")
            return []
    
    def get_data_for_analysis(self, system, phase, limit=None):
        """Get measurement data for a specific system and phase"""
        if not system or phase is None:
            return None
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build the query
            query = '''
            SELECT d.time_value, d.current_value, m.sample_interval
            FROM data_points d
            JOIN measurements m ON d.measurement_id = m.id
            JOIN systems s ON m.gpu_system_id = s.id
            WHERE s.system_type = ? AND m.phase = ?
            ORDER BY d.time_value
            '''
            
            params = [system, phase]
            
            # Add limit if specified
            if limit is not None and limit > 0:
                query += ' LIMIT ?'
                params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return None
            
            time_values = np.array([row[0] for row in results])
            current_values = np.array([row[1] for row in results])
            sample_interval = results[0][2]
            
            # Calculate additional properties
            sample_rate = 1 / sample_interval if sample_interval > 0 else 0
            
            return {
                'time': time_values,
                'current': current_values,
                'sample_interval': sample_interval,
                'sample_rate': sample_rate,
                'num_samples': len(time_values),
                'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting data for {system} phase {phase}: {e}")
            return None

    def save_data_to_csv(self, data, filename):
        """Save data to a CSV file"""
        if data is None or 'time' not in data or 'current' not in data:
            return False
            
        try:
            df = pd.DataFrame({
                'time': data['time'],
                'current': data['current']
            })
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            return False
    
    def save_data_to_database(self, data, system_type, phase):
        """Save uploaded data to the database"""
        if data is None or 'time' not in data or 'current' not in data:
            return False
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if system exists
            cursor.execute("SELECT id FROM systems WHERE system_type = ?", (system_type,))
            system_id = cursor.fetchone()
            
            if system_id:
                system_id = system_id[0]
            else:
                # Create new system
                cursor.execute(
                    "INSERT INTO systems (system_type, date, time) VALUES (?, ?, ?)",
                    (system_type, "2025-04-24", "12:00:00")  # Current date
                )
                system_id = cursor.lastrowid
            
            # Create new measurement
            cursor.execute(
                "INSERT INTO measurements (gpu_system_id, phase, sample_interval, vertical_units, record_length) VALUES (?, ?, ?, ?, ?)",
                (system_id, phase, data['sample_interval'], 'A', len(data['time']))
            )
            measurement_id = cursor.lastrowid
            
            # Insert data points in batches
            batch_size = 1000
            data_to_insert = [(measurement_id, data['time'][i], data['current'][i]) for i in range(len(data['time']))]
            
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i+batch_size]
                cursor.executemany(
                    "INSERT INTO data_points (measurement_id, time_value, current_value) VALUES (?, ?, ?)", 
                    batch
                )
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            return False