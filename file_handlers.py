import os
import pandas as pd
import numpy as np
import io
import re
import logging
import struct
import base64
import sqlite3
from config import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

class FileHandler:
    @staticmethod
    def load_data_from_csv(file_path):
        """Load data from a CSV file"""
        try:
            df = pd.read_csv(file_path)
            if 'time' in df.columns and 'current' in df.columns:
                time_values = df['time'].values
                current_values = df['current'].values
                
                if len(time_values) > 1:
                    sample_interval = time_values[1] - time_values[0]
                else:
                    sample_interval = 0.001  # Default
                
                sample_rate = 1 / sample_interval if sample_interval > 0 else 0
                
                return {
                    'time': time_values,
                    'current': current_values,
                    'sample_interval': sample_interval,
                    'sample_rate': sample_rate,
                    'num_samples': len(time_values),
                    'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
                }
            else:
                logger.warning("CSV file must contain 'time' and 'current' columns")
                return None
        except Exception as e:
            logger.error(f"Error loading data from CSV: {e}")
            return None
    
    @staticmethod
    def load_data_from_excel(file_path):
        """Load data from an Excel file"""
        try:
            df = pd.read_excel(file_path)
            if 'time' in df.columns and 'current' in df.columns:
                time_values = df['time'].values
                current_values = df['current'].values
                
                if len(time_values) > 1:
                    sample_interval = time_values[1] - time_values[0]
                else:
                    sample_interval = 0.001  # Default
                
                sample_rate = 1 / sample_interval if sample_interval > 0 else 0
                
                return {
                    'time': time_values,
                    'current': current_values,
                    'sample_interval': sample_interval,
                    'sample_rate': sample_rate,
                    'num_samples': len(time_values),
                    'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
                }
            else:
                logger.warning("Excel file must contain 'time' and 'current' columns")
                return None
        except Exception as e:
            logger.error(f"Error loading data from Excel: {e}")
            return None
    
    @staticmethod
    def load_data_from_binary(file_path):
        """Load data from a binary file"""
        try:
            with open(file_path, 'rb') as f:
                # Assuming a simple binary format: [sample_interval (float)][num_samples (int)][time values][current values]
                sample_interval = struct.unpack('f', f.read(4))[0]
                num_samples = struct.unpack('i', f.read(4))[0]
                
                time_values = np.array(struct.unpack(f'{num_samples}f', f.read(4 * num_samples)))
                current_values = np.array(struct.unpack(f'{num_samples}f', f.read(4 * num_samples)))
                
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
            logger.error(f"Error loading data from binary file: {e}")
            return None
            
    @staticmethod
    def load_data_from_sqlite(file_path, table_name=None):
        """Load data from a SQLite database"""
        try:
            conn = sqlite3.connect(file_path)
            
            # First, try to get data from data_points table
            try:
                if table_name:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                else:
                    # Try several common table names
                    for possible_table in ['data_points', 'measurements', 'data', 'power_data', 'waveform_data']:
                        try:
                            # Check if table exists
                            table_check = pd.read_sql_query(
                                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{possible_table}'", 
                                conn
                            )
                            if len(table_check) > 0:
                                df = pd.read_sql_query(f"SELECT * FROM {possible_table}", conn)
                                if 'time_value' in df.columns and 'current_value' in df.columns:
                                    # Rename columns to standard format
                                    df.rename(columns={'time_value': 'time', 'current_value': 'current'}, inplace=True)
                                    break
                                elif 'time' in df.columns and 'current' in df.columns:
                                    break
                        except:
                            continue
                    else:
                        # If no table with the right columns is found
                        raise ValueError("Could not find a suitable table in the database")
            except:
                # If specific tables don't work, find any table with time and current columns
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                found_table = False
                
                for table in tables['name']:
                    try:
                        columns = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
                        column_names = columns['name'].tolist()
                        
                        # Check for time and current columns
                        time_col = next((col for col in column_names if 'time' in col.lower()), None)
                        current_col = next((col for col in column_names if 'current' in col.lower()), None)
                        
                        if time_col and current_col:
                            df = pd.read_sql_query(f"SELECT {time_col}, {current_col} FROM {table}", conn)
                            # Rename columns to standard format
                            df.rename(columns={time_col: 'time', current_col: 'current'}, inplace=True)
                            found_table = True
                            break
                    except:
                        continue
                
                if not found_table:
                    logger.warning("Could not find a table with time and current columns")
                    return None
            
            conn.close()
            
            if 'time' in df.columns and 'current' in df.columns:
                time_values = df['time'].values
                current_values = df['current'].values
                
                if len(time_values) > 1:
                    sample_interval = time_values[1] - time_values[0]
                else:
                    sample_interval = 0.001  # Default
                
                sample_rate = 1 / sample_interval if sample_interval > 0 else 0
                
                return {
                    'time': time_values,
                    'current': current_values,
                    'sample_interval': sample_interval,
                    'sample_rate': sample_rate,
                    'num_samples': len(time_values),
                    'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
                }
            else:
                logger.warning("Database must contain 'time' and 'current' columns")
                return None
        except Exception as e:
            logger.error(f"Error loading data from SQLite database: {e}")
            return None
    
    @staticmethod
    def parse_uploaded_file(contents, filename, format_type=None):
        """Parse uploaded file contents based on file format"""
        if contents is None:
            return None
            
        # Extract content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            # Get file extension
            file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            # If format_type isn't specified, try to infer from extension
            if format_type is None:
                if file_extension == 'csv':
                    format_type = 'csv'
                elif file_extension in ['xlsx', 'xls']:
                    format_type = 'excel'
                elif file_extension == 'bin':
                    format_type = 'binary'
                elif file_extension == 'db':
                    format_type = 'sqlite'
                else:
                    logger.warning(f"Unknown file extension: {file_extension}")
                    return None
            
            # Handle different file formats
            if format_type == 'sqlite':
                # Save the database file temporarily
                temp_db_path = 'temp_uploaded.db'
                with open(temp_db_path, 'wb') as f:
                    f.write(decoded)
                data = FileHandler.load_data_from_sqlite(temp_db_path)
                # Clean up temp file
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
                return data
            elif format_type == 'csv':
                return FileHandler._parse_csv(decoded)
            elif format_type == 'excel':
                return FileHandler._parse_excel(decoded)
            elif format_type == 'binary':
                return FileHandler._parse_binary(decoded)
            else:
                logger.warning(f"Unsupported format type: {format_type}")
                return None
        except Exception as e:
            logger.error(f"Error parsing file {filename}: {e}")
            return None
    
    @staticmethod
    def _parse_csv(decoded):
        """Parse CSV data"""
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if 'time' in df.columns and 'current' in df.columns:
                time_values = df['time'].values
                current_values = df['current'].values
                
                if len(time_values) > 1:
                    sample_interval = time_values[1] - time_values[0]
                else:
                    sample_interval = 0.001  # Default
                
                sample_rate = 1 / sample_interval if sample_interval > 0 else 0
                
                return {
                    'time': time_values,
                    'current': current_values,
                    'sample_interval': sample_interval,
                    'sample_rate': sample_rate,
                    'num_samples': len(time_values),
                    'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
                }
            else:
                logger.warning("CSV file must contain 'time' and 'current' columns")
                return None
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return None
    
    @staticmethod
    def _parse_excel(decoded):
        """Parse Excel data"""
        try:
            df = pd.read_excel(io.BytesIO(decoded))
            if 'time' in df.columns and 'current' in df.columns:
                time_values = df['time'].values
                current_values = df['current'].values
                
                if len(time_values) > 1:
                    sample_interval = time_values[1] - time_values[0]
                else:
                    sample_interval = 0.001  # Default
                
                sample_rate = 1 / sample_interval if sample_interval > 0 else 0
                
                return {
                    'time': time_values,
                    'current': current_values,
                    'sample_interval': sample_interval,
                    'sample_rate': sample_rate,
                    'num_samples': len(time_values),
                    'duration': time_values[-1] - time_values[0] if len(time_values) > 0 else 0
                }
            else:
                logger.warning("Excel file must contain 'time' and 'current' columns")
                return None
        except Exception as e:
            logger.error(f"Error parsing Excel: {e}")
            return None
    
    @staticmethod
    def _parse_binary(decoded):
        """Parse binary data"""
        try:
            file_data = io.BytesIO(decoded)
            sample_interval = struct.unpack('f', file_data.read(4))[0]
            num_samples = struct.unpack('i', file_data.read(4))[0]
            
            time_values = np.array(struct.unpack(f'{num_samples}f', file_data.read(4 * num_samples)))
            current_values = np.array(struct.unpack(f'{num_samples}f', file_data.read(4 * num_samples)))
            
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
            logger.error(f"Error parsing binary: {e}")
            return None