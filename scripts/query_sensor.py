#!/usr/bin/env python3

import argparse
import requests
from zeroconf import Zeroconf, ServiceBrowser
import time
import socket

class SensorListener:
    def __init__(self):
        self.sensor_url = None
    
    def remove_service(self, zeroconf, type, name):
        pass
    
    def add_service(self, zeroconf, type, name):
        if "Temperature Sensor 2" in name:
            info = zeroconf.get_service_info(type, name)
            if info:
                addr = socket.inet_ntoa(info.addresses[0])
                port = info.port
                self.sensor_url = f"http://{addr}:{port}"
    
    def update_service(self, zeroconf, type, name):
        pass

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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Query temperature sensor')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-t', '--temperature', action='store_true', help='Get temperature')
    group.add_argument('-w', '--wifi', action='store_true', help='Get WiFi strength')
    args = parser.parse_args()

    # Find sensor
    zeroconf = Zeroconf()
    listener = SensorListener()
    browser = ServiceBrowser(zeroconf, "_temperature._tcp.local.", listener)

    # Wait for sensor discovery
    print("Searching for Temperature Sensor 2...")
    time.sleep(2)

    if not listener.sensor_url:
        print("Sensor not found!")
        zeroconf.close()
        return 1

    # Query sensor based on argument
    if args.temperature:
        temp = get_temperature(listener.sensor_url)
        if temp is not None:
            print(f"Temperature: {temp:.2f}°C")
        else:
            print("Error reading temperature")
    
    elif args.wifi:
        rssi = get_wifi_strength(listener.sensor_url)
        if rssi is not None:
            print(f"WiFi Signal Strength: {rssi} dBm")
            # Add signal quality indication
            if rssi >= -50:
                print("Signal Quality: Excellent")
            elif rssi >= -60:
                print("Signal Quality: Good")
            elif rssi >= -70:
                print("Signal Quality: Fair")
            else:
                print("Signal Quality: Poor")
        else:
            print("Error reading WiFi strength")

    zeroconf.close()
    return 0

if __name__ == "__main__":
    exit(main())