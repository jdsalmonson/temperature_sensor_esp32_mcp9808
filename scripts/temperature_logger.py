from collections import OrderedDict
import requests
import time
from datetime import datetime
import csv
import sys
from zeroconf import ServiceBrowser, Zeroconf
import socket

class TemperatureSensorListener:
    def __init__(self):
        self.sensors = OrderedDict()
        self.expected_sensors = set()  # Keep track of previously seen sensors
    
    def _sort_sensors(self):
        """Helper method to sort sensors by name"""
        self.sensors = OrderedDict(sorted(self.sensors.items(), key=lambda x: x[0]))
    
    def remove_service(self, zeroconf, type, name):
        print(f"Warning: Service {name} reported as removed")
        # Don't remove the sensor, just log the warning
        # The sensor will be marked as ERROR in readings if unreachable
    
    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addr = socket.inet_ntoa(info.addresses[0])
            port = info.port
            self.sensors[name] = f"http://{addr}:{port}"
            self.expected_sensors.add(name)  # Add to expected sensors
            print(f"Found sensor at {self.sensors[name]}")
            self._sort_sensors()
    
    def update_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and name in self.sensors:
            addr = socket.inet_ntoa(info.addresses[0])
            port = info.port
            new_url = f"http://{addr}:{port}"
            if new_url != self.sensors[name]:
                print(f"Service {name} updated from {self.sensors[name]} to {new_url}")
                self.sensors[name] = new_url
                self._sort_sensors()

def get_temperature(url):
    max_retries = 3
    retry_delay = 10  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{url}/temperature", timeout=5)
            if response.status_code == 200:
                return float(response.text)
            else:
                print(f"Error: HTTP {response.status_code}, attempt {attempt + 1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}, attempt {attempt + 1}/{max_retries}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    return None

def initialize_sensors():
    """Initialize and return zeroconf and listener objects"""
    zeroconf = Zeroconf()
    listener = TemperatureSensorListener()
    browser = ServiceBrowser(zeroconf, "_temperature._tcp.local.", listener)
    
    print("Searching for temperature sensors...")
    time.sleep(2)  # Give some time for discovery
    
    return zeroconf, listener, browser

def main():
    OUTPUT_FILE = "temperature_log.csv"
    INTERVAL = 5  # seconds
    MAX_MISSING_DATA = 5  # Maximum consecutive missing data points before reinitializing
    REINIT_WAIT = 60  # seconds to wait before reinitializing

    # Initialize zeroconf
    zeroconf, listener, browser = initialize_sensors()

    if not listener.sensors:
        print("No sensors found!")
        zeroconf.close()
        return

    print(f"Starting temperature logging to {OUTPUT_FILE}")
    print("Press Ctrl+C to stop")
    
    # Create/open CSV file with header
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            header = ['Timestamp'] + list(listener.sensors.keys())
            writer.writerow(header)
    
    missing_data_count = 0
    
    try:
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            readings = [timestamp]
            all_errors = True  # Track if all readings are errors
            
            for sensor_name, sensor_url in listener.sensors.items():
                temp = get_temperature(sensor_url)
                readings.append(temp if temp is not None else 'ERROR')
                print(f"{timestamp} - {sensor_name}: {temp if temp is not None else 'ERROR'}°C")
                if temp is not None:
                    all_errors = False
            
            # Write to CSV
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(readings)
            
            # Check if we need to reinitialize
            if all_errors:
                missing_data_count += 1
                print(f"Warning: Missing data count: {missing_data_count}/{MAX_MISSING_DATA}")
                
                if missing_data_count >= MAX_MISSING_DATA:
                    print(f"Too many missing readings. Reinitializing in {REINIT_WAIT} seconds...")
                    time.sleep(REINIT_WAIT)
                    
                    # Close existing zeroconf
                    zeroconf.close()
                    
                    # Reinitialize
                    print("Reinitializing sensor discovery...")
                    zeroconf, listener, browser = initialize_sensors()
                    
                    # Reset counter
                    missing_data_count = 0
                    
                    # If still no sensors, wait and try again
                    if not listener.sensors:
                        print("No sensors found after reinitialization. Waiting to retry...")
                        continue
                    
                    print("Reinitialization complete. Resuming logging...")
            else:
                missing_data_count = 0  # Reset counter if we get any valid reading
            
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        print("\nLogging stopped by user")
    finally:
        zeroconf.close()

if __name__ == "__main__":
    main()