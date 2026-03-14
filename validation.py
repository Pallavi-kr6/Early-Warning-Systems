def validate_flood_inputs(rainfall, river_level, soil_moisture):
    """Validate flood inputs and check logical consistency"""
    # Check ranges
    if not (0 <= rainfall <= 300):
        return False, "Rainfall must be between 0 and 300 mm"
    if not (0 <= river_level <= 20):
        return False, "River level must be between 0 and 20 m"
    if not (0 <= soil_moisture <= 100):
        return False, "Soil moisture must be between 0 and 100%"

    # Check negative values
    if rainfall < 0 or river_level < 0 or soil_moisture < 0:
        return False, "All values must be non-negative"

    # Logical inconsistencies
    if rainfall > 150 and river_level == 0:
        return False, "Heavy rainfall cannot occur with zero river level"
    if rainfall > 120 and soil_moisture < 10:
        return False, "Heavy rainfall should increase soil moisture"
    if soil_moisture > 80 and rainfall < 5:
        return False, "High soil moisture requires some rainfall"
    if rainfall == 0 and river_level > 15:
        return False, "Zero rainfall cannot cause very high river levels"

    return True, ""

def validate_earthquake_inputs(seismic_activity, ground_displacement, fault_distance, previous_earthquakes):
    """Validate earthquake inputs and check logical consistency"""
    # Check ranges
    if not (0 <= seismic_activity <= 10):
        return False, "Seismic activity must be between 0 and 10"
    if not (0 <= ground_displacement <= 100):
        return False, "Ground displacement must be between 0 and 100 mm"
    if not (0 <= fault_distance <= 500):
        return False, "Fault distance must be between 0 and 500 km"
    if not (0 <= previous_earthquakes <= 100):
        return False, "Previous earthquakes must be between 0 and 100"

    # Check negative values
    if any(val < 0 for val in [seismic_activity, ground_displacement, fault_distance, previous_earthquakes]):
        return False, "All values must be non-negative"

    # Logical inconsistencies
    if seismic_activity > 7 and ground_displacement == 0:
        return False, "High seismic activity should cause ground displacement"
    if seismic_activity < 1 and ground_displacement > 50:
        return False, "Low seismic activity cannot cause high ground displacement"
    if fault_distance > 400 and seismic_activity > 8:
        return False, "High seismic activity unlikely at very distant faults"

    return True, ""

def validate_heatwave_inputs(max_temp, min_temp, humidity, wind_speed):
    """Validate heatwave inputs and check logical consistency"""
    # Check ranges
    if not (20 <= max_temp <= 55):
        return False, "Max temperature must be between 20 and 55°C"
    if not (10 <= min_temp <= 40):
        return False, "Min temperature must be between 10 and 40°C"
    if not (0 <= humidity <= 100):
        return False, "Humidity must be between 0 and 100%"
    if not (0 <= wind_speed <= 30):
        return False, "Wind speed must be between 0 and 30 km/h"

    # Check negative values
    if any(val < 0 for val in [max_temp, min_temp, humidity, wind_speed]):
        return False, "All values must be non-negative"

    # Logical inconsistencies
    if min_temp > max_temp:
        return False, "Min temperature cannot be higher than max temperature"
    if max_temp < 25 and humidity > 80:
        return False, "Low temperatures with very high humidity are unusual"
    if max_temp > 45 and humidity < 5:
        return False, "Very high temperatures with very low humidity are unusual"

    return True, ""