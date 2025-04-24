import os
import re
import sqlite3

def examine_file(filename):
    """Print sample content from the CSV file"""
    print(f"Examining {filename}...")
    try:
        with open(filename, 'r') as f:
            # Read first 10 lines
            print("First 10 lines:")
            for i in range(10):
                line = f.readline().strip()
                if not line:
                    break
                print(f"  {i}: {line}")
            
            # Read a sample from the middle
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(file_size // 2)
            f.readline()  # Skip partial line
            
            print("\nSample from middle:")
            for i in range(5):
                line = f.readline().strip()
                if not line:
                    break
                print(f"  {line}")
                
            # Count lines in file
            f.seek(0)
            line_count = sum(1 for _ in f)
            print(f"\nTotal lines in file: {line_count}")
            
            return line_count
    except Exception as e:
        print(f"Error examining file: {e}")
        return 0

def extract_metadata(file_path):
    """Extract metadata from the CSV file header"""
    metadata = {
        'sample_interval': 0.0001,
        'record_length': 5000000,
        'vertical_units': 'A',
        'header_lines': 10  # Default number of header lines
    }
    
    try:
        with open(file_path, 'r') as f:
            for i in range(20):  # Check up to 20 lines for metadata
                line = f.readline().strip()
                if not line:
                    continue
                
                # Check for specific metadata fields
                if 'Sample Interval' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            metadata['sample_interval'] = float(parts[1])
                        except ValueError:
                            pass
                
                elif 'Record Length' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            metadata['record_length'] = int(parts[1])
                        except ValueError:
                            pass
                
                elif 'Vertical Units' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        metadata['vertical_units'] = parts[1]
                        
                # Look for data start - check if this looks like data
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            float(parts[0])
                            float(parts[1])
                            metadata['header_lines'] = i  # This is where data starts
                            break
                        except ValueError:
                            pass
                            
                # Special case: if we see a line with "Labels" it's probably right before data
                if line.startswith("Labels"):
                    metadata['header_lines'] = i + 1
                    break
    
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    
    return metadata

def read_file_simple(file_path, db_path="power_data.db"):
    """Read CSV file with specific format for oscilloscope data"""
    filename = os.path.basename(file_path)
    
    # Extract metadata from filename
    match = re.match(r'Scope_(\d+)_(\d+)_phase(\d)\.csv', filename)
    if not match:
        print(f"Skipping {filename} - doesn't match expected pattern")
        return 0
    
    date_str, time_str, phase = match.groups()
    phase = int(phase)
    
    # Determine system type
    system_type = "Unknown"
    combined = f"{date_str}_{time_str}"
    if "250228_1530" in combined:
        system_type = "GB200_1"
    elif "250228_1431" in combined:
        system_type = "GB200_2"
    elif "250227_2042" in combined:
        system_type = "NVL36"
    
    # Extract metadata from file header
    metadata = extract_metadata(file_path)
    sample_interval = metadata['sample_interval']
    
    # Connect to database
    conn = sqlite3.connect(db_path)
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
    
    # Insert system info
    cursor.execute('''
    INSERT OR IGNORE INTO systems (system_type, date, time)
    VALUES (?, ?, ?)
    ''', (system_type, date_str, time_str))
    
    # Get system ID
    cursor.execute('''
    SELECT id FROM systems WHERE system_type=? AND date=? AND time=?
    ''', (system_type, date_str, time_str))
    system_id = cursor.fetchone()[0]
    
    # Insert measurement
    cursor.execute('''
    INSERT INTO measurements (gpu_system_id, phase, sample_interval, vertical_units, record_length)
    VALUES (?, ?, ?, ?, ?)
    ''', (system_id, phase, metadata['sample_interval'], metadata['vertical_units'], metadata['record_length']))
    
    measurement_id = cursor.lastrowid
    
    # Process file content
    print(f"Processing {filename}...")
    
    # Now read the file and insert data
    total_rows = 0
    
    try:
        with open(file_path, 'r') as f:
            # Skip header lines
            header_lines = metadata['header_lines']
            for _ in range(header_lines):
                f.readline()
            
            # Look for line with column names (if it exists)
            line = f.readline().strip()
            if line and not line[0].isdigit() and ',' in line:
                # This is probably a header line with column names, not data
                column_names = line.split(',')
                print(f"Found column names: {column_names}")
            else:
                # This is data, go back one line
                f.seek(f.tell() - len(line) - 1)
            
            # Read data in batches
            batch = []
            batch_size = 10000
            line_count = 0
            
            print("Starting to process data...")
            
            for line_num, line in enumerate(f):
                if line_num % 500000 == 0 and line_num > 0:
                    print(f"  Processed {line_num} lines...")
                
                line = line.strip()
                if not line:
                    continue
                    
                # Split by comma
                parts = line.split(',')
                if len(parts) < 2:
                    continue  # Skip lines with insufficient data
                    
                try:
                    # Parse time and current values
                    time_value = float(parts[0])
                    current_value = float(parts[1])
                    
                    batch.append((measurement_id, time_value, current_value))
                    line_count += 1
                    
                    # Insert in batches
                    if len(batch) >= batch_size:
                        cursor.executemany('''
                        INSERT INTO data_points (measurement_id, time_value, current_value)
                        VALUES (?, ?, ?)
                        ''', batch)
                        conn.commit()
                        total_rows += len(batch)
                        batch = []
                        
                except Exception as e:
                    if line_count < 10:  # Only show first few errors
                        print(f"Error processing line: {line} - {e}")
            
            # Insert remaining rows
            if batch:
                cursor.executemany('''
                INSERT INTO data_points (measurement_id, time_value, current_value)
                VALUES (?, ?, ?)
                ''', batch)
                conn.commit()
                total_rows += len(batch)
        
    except Exception as e:
        print(f"Error processing file data: {e}")
    
    print(f"Processed {total_rows} data points from {filename}")
    conn.close()
    return total_rows

def main():
    # Process all CSV files in current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # First, examine one file to understand its structure
    if csv_files:
        examine_file(csv_files[0])
    
    # Process each file
    for i, filename in enumerate(csv_files):
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {filename}")
        read_file_simple(filename)
    
    print("\nAll files processed. Run power_dashboard.py to visualize the data.")

if __name__ == "__main__":
    main()