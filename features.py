def engineer_flood_features(rainfall, river_level, soil_moisture):
    """Add engineered features for flood prediction"""
    flood_risk_factor = rainfall * soil_moisture / 100
    return [rainfall, river_level, soil_moisture, flood_risk_factor]

def engineer_earthquake_features(seismic_activity, ground_displacement, fault_distance, previous_earthquakes):
    """Return raw features for earthquake prediction"""
    return [seismic_activity, ground_displacement, fault_distance, previous_earthquakes]

def engineer_heatwave_features(max_temp, min_temp, humidity, wind_speed):
    """Return raw features for heatwave prediction."""
    return [max_temp, min_temp, humidity, wind_speed]