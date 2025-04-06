#!/usr/bin/env python3
"""
One can query the sensors using curl if you know the IP address of the ESP32:
curl http://<esp32-ip>/temperature
curl http://<esp32-ip>/rssi
"""

from zeroconf import Zeroconf, ServiceBrowser
import requests
import time
import socket
from collections import OrderedDict

class SensorListener:
    def __init__(self):
        self.sensors = OrderedDict()  # Store sensors in order
    
    def remove_service(self, zeroconf, type, name):
        if name in self.sensors:
            print(f"Service {name} removed")
            del self.sensors[name]
    
    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            addr = socket.inet_ntoa(info.addresses[0])
            port = info.port
            self.sensors[name] = f"http://{addr}:{port}"
            print(f"Found sensor: {name}")
    
    def update_service(self, zeroconf, type, name):
        self.add_service(zeroconf, type, name)

def get_temperature(url):
    try:
        response = requests.get(f"{url}/temperature", timeout=5)
        if response.status_code == 200:
            return float(response.text)
        return None
    except:
        return None

def get_wifi_strength(url):
    try:
        response = requests.get(f"{url}/rssi", timeout=5)
        if response.status_code == 200:
            return int(response.text)
        return None
    except:
        return None

def get_signal_quality(rssi):
    if rssi >= -50:
        return "Excellent"
    elif rssi >= -60:
        return "Good"
    elif rssi >= -70:
        return "Fair"
    else:
        return "Poor"

def main():
    # Initialize zeroconf
    zeroconf = Zeroconf()
    listener = SensorListener()
    browser = ServiceBrowser(zeroconf, "_temperature._tcp.local.", listener)

    # Wait for sensor discovery
    print("Searching for temperature sensors...")
    time.sleep(2)  # Give some time for discovery

    if not listener.sensors:
        print("No sensors found!")
        zeroconf.close()
        return 1

    # Sort sensors by name to ensure consistent order
    sorted_sensors = OrderedDict(sorted(listener.sensors.items()))

    # Query each sensor
    print("\nSensor Readings:")
    print("-" * 50)
    
    for name, url in sorted_sensors.items():
        print(f"\nSensor: {name}")
        print("-" * 20)
        
        # Get temperature
        temp = get_temperature(url)
        if temp is not None:
            print(f"Temperature: {temp:.2f}°C ({temp * 9/5 + 32:.2f}°F)")
        else:
            print("Temperature: ERROR")
        
        # Get WiFi strength
        rssi = get_wifi_strength(url)
        if rssi is not None:
            quality = get_signal_quality(rssi)
            print(f"WiFi Signal: {rssi} dBm ({quality})")
        else:
            print("WiFi Signal: ERROR")

    print("\n" + "-" * 50)
    zeroconf.close()
    return 0

if __name__ == "__main__":
    exit(main())