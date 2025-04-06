from horemheb.config import AnalysisConfig
from horemheb.loader import load_temperature_data
from horemheb.segments import process_segments
from horemheb.analysis import analyze_segments

from pathlib import Path

# Configure analysis parameters
config = AnalysisConfig(
    delay_time=120,
    before_sunrise_delta_minutes=0,
    resample_minutes=5
)

file = Path.home() / 'Work/Animata/Esp_projects/temperature_sensor_esp32_mcp9808/esp32_wifi/notebooks' / 'temperature_logB_12.csv'
# Load and process data
df, df_resampled1, df_resampled2 = load_temperature_data(file, config)

# Find and process segments
segments = process_segments(df_resampled1, df_resampled2, config)

# Perform analysis
results = analyze_segments(segments, config)
print("Old Door")
print(f"A = {results.A:.3f} ± {results.A_err:.3f} °C/(°C/hour)")
print(f"b = {results.b:.3f} ± {results.b_err:.3f} °C")
print(f"A0 = {results.A0:.3f} ± {results.A0_err:.3f} °C/(°C/hour)")
print(f"R-squared = {results.r_squared:.3f}")
print(f"R-squared (through origin) = {results.r_squared0:.3f}")