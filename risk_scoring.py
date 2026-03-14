"""Risk scoring utilities for Disaster Alert System.

These functions compute a normalized risk score (0.0-1.0) based on multiple
input factors. The scores are designed to avoid extreme predictions driven by a
single input value.
"""

from typing import Tuple


def calculate_flood_risk(rainfall: float, river_level: float, soil_moisture: float) -> Tuple[float, str]:
    """Calculate flood risk percentage and explanation.

    Returns:
        (risk_percentage, explanation)
    """
    # Normalize inputs
    rainfall_norm = min(max(rainfall / 200.0, 0.0), 1.0)
    river_norm = min(max(river_level / 10.0, 0.0), 1.0)
    soil_norm = min(max(soil_moisture / 100.0, 0.0), 1.0)

    # Weighted score
    score = 0.4 * rainfall_norm + 0.4 * river_norm + 0.2 * soil_norm
    risk_percentage = round(max(0.0, min(score, 1.0)) * 100, 2)

    parts = []
    if rainfall_norm > 0.7:
        parts.append("heavy rainfall")
    if river_norm > 0.7:
        parts.append("high river level")
    if soil_norm > 0.7:
        parts.append("saturated soil")

    if parts:
        explanation = " and ".join(parts).capitalize() + " increase flood risk."
    else:
        explanation = "Conditions are generally moderate for flood risk."

    return risk_percentage, explanation


def calculate_earthquake_risk(seismic_activity: float,
                              ground_displacement: float,
                              fault_distance: float,
                              previous_earthquakes: int) -> Tuple[float, str]:
    """Calculate earthquake risk percentage and explanation."""
    # Updated scoring with specified weights
    seismic_score = seismic_activity / 10.0
    displacement_score = ground_displacement / 100.0
    distance_score = 1 - (fault_distance / 500.0)
    history_score = previous_earthquakes / 50.0

    risk_score = (0.5 * seismic_score +
                  0.25 * displacement_score +
                  0.15 * distance_score +
                  0.1 * history_score)

    risk_percentage = min(max(risk_score * 100, 0), 100)

    # Classify risk
    if risk_percentage >= 70:
        risk_level = "high"
        explanation = "High seismic activity and ground displacement indicate significant earthquake risk."
    elif risk_percentage >= 40:
        risk_level = "medium"
        explanation = "Moderate seismic conditions suggest earthquake risk."
    else:
        risk_level = "low"
        explanation = "Seismic activity is within normal ranges."

    return round(risk_percentage, 2), explanation


def calculate_heatwave_risk(max_temp: float, min_temp: float, humidity: float, wind_speed: float) -> Tuple[float, str]:
    """Calculate heatwave risk percentage and explanation using the same rules as dataset generation."""
    # Determine risk level based on temperature and humidity rules
    if max_temp >= 44 or (max_temp >= 40 and min_temp >= 30):
        risk_level = "high"
        risk_percentage = 85.0  # High risk percentage
        explanation = "Extreme temperatures indicate severe heatwave conditions."
    elif 38 <= max_temp < 44 or (max_temp >= 36 and humidity >= 60):
        risk_level = "medium"
        risk_percentage = 55.0  # Medium risk percentage
        explanation = "High temperatures suggest moderate heatwave risk."
    else:
        risk_level = "low"
        risk_percentage = 25.0  # Low risk percentage
        explanation = "Temperatures are within normal ranges."

    return risk_percentage, explanation
