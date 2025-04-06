from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    delay_time: int = 120  # minutes
    before_sunrise_delta_minutes: int = 0
    resample_minutes: int = 5
