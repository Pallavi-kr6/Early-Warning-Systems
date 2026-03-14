from flask import Flask, render_template, request, send_file
import numpy as np
import joblib
import json
import os
import datetime
import requests
import tempfile
from fpdf import FPDF
from validation import validate_flood_inputs, validate_earthquake_inputs, validate_heatwave_inputs
from features import engineer_flood_features, engineer_earthquake_features, engineer_heatwave_features
from risk_scoring import calculate_flood_risk, calculate_earthquake_risk, calculate_heatwave_risk

app = Flask(__name__)

# Load models and scalers
models_loaded = False
heat_model = earthquake_model = flood_model = None
flood_scaler = heat_scaler = earthquake_scaler = None

try:
    flood_model = joblib.load('models/flood_model.pkl')
    earthquake_model = joblib.load('models/earthquake_model.pkl')
    heat_model = joblib.load('models/heatwave_model.pkl')
    flood_scaler = joblib.load('models/flood_scaler.pkl')
    heat_scaler = joblib.load('models/heatwave_scaler.pkl')
    earthquake_scaler = joblib.load('models/earthquake_scaler.pkl')
    models_loaded = True
    print("SUCCESS: Models loaded successfully")
except Exception as e:
    print(f"ERROR: Error loading models: {e}")
    models_loaded = False

# Load city coordinates and helplines
with open('data/city_coordinates.json') as f:
    CITY_COORDINATES = json.load(f)

with open('data/emergency_helplines.json') as f:
    EMERGENCY_HELPLINES = json.load(f)

# Google Maps API Key - Replace with your actual key
GMAPS_API_KEY = "AIzaSyDeuGhbyL2Atc_nKo8ZYhx8LwWL0QYlIOo"

# Alert generation function with earthquake support
def generate_alert(disaster_type, severity, location):
    """Generate customized alerts for different stakeholders with location"""
    alerts = {
        # Government agencies
        'government': {
            'flood': {
                'high': f"URGENT: Severe flooding predicted in {location}. Immediate evacuation needed. Deploy emergency response teams and resources. [City: {location}]",
                'medium': f"WARNING: Moderate flooding expected in {location}. Prepare emergency shelters and response teams. [City: {location}]",
                'low': f"ADVISORY: Minor flooding possible in {location}. Monitor situation and prepare response resources. [City: {location}]"
            },
            'heatwave': {
                'high': f"URGENT: Extreme heat wave predicted in {location}. Activate cooling centers and emergency medical services. [City: {location}]",
                'medium': f"WARNING: Significant heat wave expected in {location}. Prepare public cooling facilities and check on vulnerable populations. [City: {location}]",
                'low': f"ADVISORY: Mild heat wave possible in {location}. Prepare for increased cooling needs and public advisories. [City: {location}]"
            },
            'earthquake': {
                'high': f"URGENT: High earthquake risk detected in {location}. Activate emergency response protocols. Deploy search and rescue teams. [City: {location}]",
                'medium': f"WARNING: Moderate earthquake risk in {location}. Prepare emergency response teams and check building safety protocols. [City: {location}]",
                'low': f"ADVISORY: Low earthquake risk detected in {location}. Review emergency protocols and building safety measures. [City: {location}]"
            }
        },

        # NGOs
        'ngo': {
            'flood': {
                'high': f"Urgent assistance needed: Severe flooding predicted in {location}. Prepare relief supplies, medical teams, and temporary shelters. [City: {location}]",
                'medium': f"Alert: Moderate flooding expected in {location}. Ready relief supplies and volunteer teams. [City: {location}]",
                'low': f"Notice: Minor flooding possible in {location}. Monitor situation and be prepared to assist if needed. [City: {location}]"
            },
            'heatwave': {
                'high': f"Urgent assistance needed: Extreme heat wave predicted in {location}. Prepare water distribution, cooling stations, and medical aid. [City: {location}]",
                'medium': f"Alert: Significant heat wave expected in {location}. Prepare water supplies and check on elderly and vulnerable populations. [City: {location}]",
                'low': f"Notice: Mild heat wave possible in {location}. Consider preparing heat relief measures. [City: {location}]"
            },
            'earthquake': {
                'high': f"Urgent assistance needed: High earthquake risk in {location}. Prepare emergency medical supplies, search and rescue equipment, and temporary shelters. [City: {location}]",
                'medium': f"Alert: Moderate earthquake risk in {location}. Ready emergency supplies and volunteer response teams. [City: {location}]",
                'low': f"Notice: Low earthquake risk in {location}. Review emergency preparedness and supply readiness. [City: {location}]"
            }
        },

        # Public
        'public': {
            'flood': {
                'high': f"EMERGENCY ALERT: Severe flooding expected in {location}. Evacuate immediately to higher ground. Follow official instructions. [City: {location}]",
                'medium': f"FLOOD WARNING: Significant flooding possible in {location}. Prepare emergency supplies and be ready to evacuate if instructed. [City: {location}]",
                'low': f"FLOOD WATCH: Minor flooding possible in {location}. Stay informed and prepare emergency supplies. [City: {location}]"
            },
            'heatwave': {
                'high': f"EMERGENCY ALERT: Dangerous heat wave expected in {location}. Stay indoors, drink plenty of water, and seek cool environments. [City: {location}]",
                'medium': f"HEAT WARNING: High temperatures expected in {location}. Limit outdoor activities, stay hydrated, and check on vulnerable neighbors. [City: {location}]",
                'low': f"HEAT ADVISORY: Warm temperatures expected in {location}. Stay hydrated and take breaks from the heat. [City: {location}]"
            },
            'earthquake': {
                'high': f"EMERGENCY ALERT: High earthquake risk in {location}. Secure heavy objects, identify safe spots, and be ready to Drop, Cover, and Hold On. [City: {location}]",
                'medium': f"EARTHQUAKE WARNING: Moderate seismic activity possible in {location}. Review earthquake safety plans and secure loose items. [City: {location}]",
                'low': f"EARTHQUAKE ADVISORY: Low seismic risk detected in {location}. Review earthquake preparedness and safety procedures. [City: {location}]"
            }
        }
    }

    return {
        'government': alerts['government'][disaster_type][severity],
        'ngo': alerts['ngo'][disaster_type][severity],
        'public': alerts['public'][disaster_type][severity]
    }

# PDF generation class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "DISASTER MANAGEMENT AUTHORITY", ln=True, align="C")
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f"OFFICIAL {self.disaster_type.upper()} {self.severity.upper()} ALERT", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, "This is an automatically generated alert from the Disaster Early Warning System.", ln=True, align="C")

# PDF generation function with earthquake support
def generate_alert_pdf(disaster_type, severity, location, data):
    try:
        # Create directory if it doesn't exist
        os.makedirs('generated_pdfs', exist_ok=True)
        
        pdf = PDF()
        pdf.disaster_type = disaster_type
        pdf.severity = severity
        pdf.add_page()

        # Map Configuration
        try:
            lat, lon = CITY_COORDINATES[location]
            map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=12&size=600x300&maptype=roadmap&markers=color:red%7C{lat},{lon}&key=AIzaSyDeuGhbyL2Atc_nKo8ZYhx8LwWL0QYlIOo"
            response = requests.get(map_url)

            if response.status_code == 200:
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    tmpfile.write(response.content)
                    img_path = tmpfile.name

                # Add image to PDF and cleanup
                pdf.image(img_path, x=10, y=40, w=180)
                os.unlink(img_path)
                pdf.ln(120)
            else:
                pdf.cell(0, 10, f"Map unavailable (Status: {response.status_code})", ln=True)
        except KeyError:
            pdf.cell(0, 10, f"Coordinates not found for {location}", ln=True)
        except Exception as map_error:
            pdf.cell(0, 10, f"Map error: {str(map_error)}", ln=True)

        # Rest of PDF content
        pdf.set_font("Arial", "", 10)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 10, f"Issued on: {current_time}", ln=True)

        # Alert Details
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"LOCATION: {location}", ln=True)
        pdf.cell(0, 10, "ALERT DETAILS:", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Data display
        pdf.set_font("Arial", "", 12)
        if disaster_type.lower() == 'flood':
            pdf.cell(0, 10, f"Rainfall: {data['rainfall']:.1f} mm", ln=True)
            pdf.cell(0, 10, f"River Level: {data['river_level']:.1f} m", ln=True)
            pdf.cell(0, 10, f"Soil Moisture: {data['soil_moisture']:.1f}%", ln=True)
        elif disaster_type.lower() == 'heatwave':
            pdf.cell(0, 10, f"Maximum Temperature: {data['max_temp']:.1f}°C", ln=True)
            pdf.cell(0, 10, f"Humidity: {data['humidity']:.1f}%", ln=True)
            pdf.cell(0, 10, f"Consecutive Hot Days: {data['consecutive_hot_days']}", ln=True)
        elif disaster_type.lower() == 'earthquake':
            pdf.cell(0, 10, f"Seismic Activity: {data['seismic_activity']:.2f}", ln=True)
            pdf.cell(0, 10, f"Ground Displacement: {data['ground_displacement']:.2f} mm", ln=True)
            pdf.cell(0, 10, f"Fault Distance: {data['fault_distance']:.1f} km", ln=True)
            pdf.cell(0, 10, f"Previous Earthquakes (30 days): {data['previous_earthquakes']}", ln=True)
            if 'magnitude' in data:
                pdf.cell(0, 10, f"Predicted Magnitude: {data['magnitude']:.1f}", ln=True)

        # Alert messages
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "ALERT MESSAGE:", ln=True)

        alerts = generate_alert(disaster_type.lower(), severity.lower(), location)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"For Government Agencies:\n{alerts['government']}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"For NGOs and Relief Organizations:\n{alerts['ngo']}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"For Public Distribution:\n{alerts['public']}")

        # Emergency contacts
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "EMERGENCY HELPLINES:", ln=True)
        pdf.set_font("Arial", "", 12)
        for name, number in EMERGENCY_HELPLINES.items():
            pdf.cell(0, 10, f"{name}: {number}", ln=True)

        # Safety instructions
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "SAFETY INSTRUCTIONS:", ln=True)
        safety_tips = {
            "flood": [
                "Move to higher ground immediately if instructed to evacuate",
                "Turn off electricity and gas if possible",
                "Avoid walking or driving through flood waters",
                "Follow news channels and official social media for updates",
                "Keep emergency supplies ready (food, water, medicines, documents)"
            ],
            "heatwave": [
                "Stay indoors in air-conditioned environments when possible",
                "Drink plenty of water, even if not thirsty",
                "Wear lightweight, light-colored, loose-fitting clothing",
                "Take cool showers or baths",
                "Check on elderly, sick, and those who live alone"
            ],
            "earthquake": [
                "Drop, Cover, and Hold On during shaking",
                "Stay away from windows, mirrors, and heavy objects",
                "If outdoors, move away from buildings, trees, and power lines",
                "After shaking stops, check for injuries and hazards",
                "Be prepared for aftershocks",
                "Have emergency supplies ready (water, food, flashlight, radio)"
            ]
        }
        pdf.set_font("Arial", "", 12)
        for tip in safety_tips.get(disaster_type.lower(), []):
            pdf.cell(0, 10, f"- {tip}", ln=True)

        # Save PDF
        filename = f"generated_pdfs/{disaster_type.lower()}_{severity.lower()}_{location.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename, "F")

        return filename
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

# Homepage with form
@app.route('/')
def home():
    return render_template('index.html', cities=list(CITY_COORDINATES.keys()))

# Prediction endpoint with validation and feature engineering
@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return render_template('result.html',
                              error=True,
                              error_message="Models could not be loaded. Please run train_models.py first.",
                              disaster_type=request.form.get('disaster', 'Unknown').title(),
                              city=request.form.get('city', 'Unknown'),
                              risk_detected=False,
                              prediction=0.0,
                              alerts=None,
                              severity=None,
                              pdf_path=None,
                              input_data={})

    disaster_type = request.form['disaster']
    city = request.form['city']
    generate_pdf = 'pdf' in request.form

    # FLOOD PREDICTION
    if disaster_type == 'flood':
        rainfall = float(request.form.get('rainfall', 100.0))
        river_level = float(request.form.get('river_level', 5.0))
        soil_moisture = float(request.form.get('soil_moisture', 50.0))

        # Validate inputs
        valid, error_msg = validate_flood_inputs(rainfall, river_level, soil_moisture)
        if not valid:
            return render_template('result.html',
                                  error=True,
                                  error_message=f"Invalid environmental conditions detected. {error_msg}",
                                  disaster_type='Flood',
                                  city=city,
                                  risk_detected=False,
                                  prediction=0.0,
                                  alerts=None,
                                  severity=None,
                                  pdf_path=None,
                                  input_data={'rainfall': rainfall, 'river_level': river_level, 'soil_moisture': soil_moisture})

        # Feature engineering
        features = engineer_flood_features(rainfall, river_level, soil_moisture)
        input_data = np.array([features])
        scaled_input = flood_scaler.transform(input_data)

        # Predict
        prediction_prob = flood_model.predict_proba(scaled_input)[0][1]  # Probability of flood (class 1)
        prediction_prob = max(0.0, min(prediction_prob, 1.0))

        # Rule-based risk score (multi-factor) and explanation
        rule_risk_percentage, explanation = calculate_flood_risk(rainfall, river_level, soil_moisture)

        # Combine ML probability and rule-based score to reduce single-variable extremes
        combined_prob = (prediction_prob + (rule_risk_percentage / 100.0)) / 2.0
        risk_probability = round(max(0.0, min(combined_prob, 1.0)) * 100, 2)

        # Determine risk level (LOW / MODERATE / HIGH)
        if risk_probability < 40:
            risk_level = "low"
        elif risk_probability < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Determine bar style
        progress_width = f"{risk_probability}%"
        if risk_level == "low":
            progress_color = "bg-success"
        elif risk_level == "medium":
            progress_color = "bg-warning"
        else:
            progress_color = "bg-danger"

        print("ML Predicted Probability:", prediction_prob)
        print("Rule-based Risk Percentage:", rule_risk_percentage)
        print("Combined Risk Percentage:", risk_probability)

        recommendation = "Stay informed and have emergency supplies ready."
        if risk_level == "medium":
            recommendation = "Prepare emergency supplies and monitor situation closely."
        elif risk_level == "high":
            recommendation = "Move to higher ground and follow evacuation alerts immediately."

        alerts = {
            'government': f"Flood risk in {city}: {risk_level.upper()} - {explanation}",
            'ngo': f"Flood risk in {city}: {risk_level.upper()} - Prepare relief operations",
            'public': f"Flood risk in {city}: {risk_level.upper()} - {recommendation}"
        } if risk_probability >= 40 else None

        pdf_path = None
        if generate_pdf and risk_probability >= 40:
            data = {'rainfall': rainfall, 'river_level': river_level, 'soil_moisture': soil_moisture}
            pdf_path = generate_alert_pdf('flood', risk_level, city, data)

        return render_template('result.html',
                              risk_detected=(risk_probability >= 40),
                              risk_probability=risk_probability,
                              risk_percentage=risk_probability,
                              risk_level=risk_level,
                              progress_width=progress_width,
                              progress_color=progress_color,
                              alerts=alerts,
                              severity=risk_level,
                              disaster_type='Flood',
                              pdf_path=pdf_path,
                              city=city,
                              input_data={'rainfall': rainfall, 'river_level': river_level, 'soil_moisture': soil_moisture})

    # EARTHQUAKE PREDICTION
    elif disaster_type == 'earthquake':
        seismic_activity = float(request.form.get('seismic_activity', 2.5))
        ground_displacement = float(request.form.get('ground_displacement', 0.5))
        fault_distance = float(request.form.get('fault_distance', 10.0))
        previous_earthquakes = int(request.form.get('previous_earthquakes', 2))

        # Validate inputs
        valid, error_msg = validate_earthquake_inputs(seismic_activity, ground_displacement, fault_distance, previous_earthquakes)
        if not valid:
            return render_template('result.html',
                                  error=True,
                                  error_message=f"Invalid environmental conditions detected. {error_msg}",
                                  disaster_type='Earthquake',
                                  city=city,
                                  risk_detected=False,
                                  prediction=0.0,
                                  alerts=None,
                                  severity=None,
                                  pdf_path=None,
                                  input_data={'seismic_activity': seismic_activity, 'ground_displacement': ground_displacement,
                                             'fault_distance': fault_distance, 'previous_earthquakes': previous_earthquakes})

        # Feature engineering
        features = engineer_earthquake_features(seismic_activity, ground_displacement, fault_distance, previous_earthquakes)
        print("Input features:", features)
        input_data = np.array([features])
        scaled_input = earthquake_scaler.transform(input_data)

        # Predict class (0=LOW, 1=MEDIUM, 2=HIGH)
        prediction_class = earthquake_model.predict(scaled_input)[0]

        # Map model prediction to risk level and probability
        if prediction_class == 0:
            ml_risk_level = "low"
            ml_probability = 25.0
        elif prediction_class == 1:
            ml_risk_level = "medium"
            ml_probability = 55.0
        else:  # prediction_class == 2
            ml_risk_level = "high"
            ml_probability = 85.0

        # Rule-based risk score (multi-factor) and explanation
        rule_risk_percentage, rule_explanation = calculate_earthquake_risk(
            seismic_activity, ground_displacement, fault_distance, previous_earthquakes)

        # Determine rule-based risk level
        if rule_risk_percentage >= 70:
            rule_risk_level = "high"
        elif rule_risk_percentage >= 40:
            rule_risk_level = "medium"
        else:
            rule_risk_level = "low"

        # Use ML prediction as primary, but escalate if rule-based indicates higher risk
        risk_level_order = {"low": 0, "medium": 1, "high": 2}
        ml_level_num = risk_level_order[ml_risk_level]
        rule_level_num = risk_level_order[rule_risk_level]

        if rule_level_num > ml_level_num:
            # Rule-based indicates higher risk, use rule-based
            risk_level = rule_risk_level
            risk_probability = rule_risk_percentage
            explanation = rule_explanation
        else:
            # Use ML prediction
            risk_level = ml_risk_level
            risk_probability = ml_probability
            explanation = rule_explanation  # Still use rule-based explanation

        # Determine bar style
        progress_width = f"{risk_probability}%"
        if risk_level == "low":
            progress_color = "bg-success"
        elif risk_level == "medium":
            progress_color = "bg-warning"
        else:
            progress_color = "bg-danger"

        print("ML Predicted Class:", prediction_class, f"({ml_risk_level})")
        print("ML Probability:", ml_probability)
        print("Rule-based Risk Percentage:", rule_risk_percentage)
        print("Combined Risk Percentage:", risk_probability)

        recommendation = "Review earthquake preparedness and safety procedures."
        if risk_level == "medium":
            recommendation = "Review earthquake safety plans and secure loose items."
        elif risk_level == "high":
            recommendation = "Secure heavy objects, identify safe spots, and be ready to Drop, Cover, and Hold On."

        alerts = {
            'government': f"Earthquake risk in {city}: {risk_level.upper()} - {explanation}",
            'ngo': f"Earthquake risk in {city}: {risk_level.upper()} - Prepare emergency response",
            'public': f"Earthquake risk in {city}: {risk_level.upper()} - {recommendation}"
        } if risk_probability >= 40 else None

        pdf_path = None
        if generate_pdf and risk_probability >= 40:
            data = {'seismic_activity': seismic_activity, 'ground_displacement': ground_displacement,
                   'fault_distance': fault_distance, 'previous_earthquakes': previous_earthquakes, 'magnitude': 3.0 + (risk_probability / 100.0) * 3}
            pdf_path = generate_alert_pdf('earthquake', risk_level, city, data)

        return render_template('result.html',
                              risk_detected=(risk_probability >= 40),
                              risk_probability=risk_probability,
                              risk_percentage=risk_probability,
                              risk_level=risk_level,
                              progress_width=progress_width,
                              progress_color=progress_color,
                              alerts=alerts,
                              severity=risk_level,
                              disaster_type='Earthquake',
                              pdf_path=pdf_path,
                              city=city,
                              input_data={'seismic_activity': seismic_activity, 'ground_displacement': ground_displacement,
                                         'fault_distance': fault_distance, 'previous_earthquakes': previous_earthquakes})

    # HEATWAVE PREDICTION
    elif disaster_type == 'heatwave':
        max_temp = float(request.form.get('max_temp', 35.0))
        min_temp = float(request.form.get('min_temp', 25.0))
        humidity = float(request.form.get('humidity', 50.0))
        wind_speed = float(request.form.get('wind_speed', 10.0))

        # Validate inputs
        valid, error_msg = validate_heatwave_inputs(max_temp, min_temp, humidity, wind_speed)
        if not valid:
            return render_template('result.html',
                                  error=True,
                                  error_message=f"Invalid environmental conditions detected. {error_msg}",
                                  disaster_type='Heat Wave',
                                  city=city,
                                  risk_detected=False,
                                  prediction=0.0,
                                  alerts=None,
                                  severity=None,
                                  pdf_path=None,
                                  input_data={'max_temp': max_temp, 'min_temp': min_temp, 'humidity': humidity, 'wind_speed': wind_speed})

        # Feature engineering
        features = engineer_heatwave_features(max_temp, min_temp, humidity, wind_speed)
        input_data = np.array([features])
        scaled_input = heat_scaler.transform(input_data)

        # Predict using multi-class model
        prediction = heat_model.predict(scaled_input)[0]  # 0=LOW, 1=MEDIUM, 2=HIGH
        prediction_prob = heat_model.predict_proba(scaled_input)[0]

        print("Input features:", features)
        print("Model prediction (class):", prediction)
        print("Prediction probabilities:", prediction_prob)

        # Map prediction to risk level and percentage
        if prediction == 2:
            risk_level = "high"
            risk_probability = round(prediction_prob[2] * 100, 2)
        elif prediction == 1:
            risk_level = "medium"
            risk_probability = round(prediction_prob[1] * 100, 2)
        else:
            risk_level = "low"
            risk_probability = round(prediction_prob[0] * 100, 2)

        # Rule-based risk score (multi-factor) and explanation
        rule_risk_percentage, explanation = calculate_heatwave_risk(max_temp, min_temp, humidity, wind_speed)

        # Combine ML prediction confidence with rule-based score
        combined_prob = (risk_probability + rule_risk_percentage) / 200.0  # Normalize to 0-1
        risk_probability = round(max(0.0, min(combined_prob, 1.0)) * 100, 2)

        # Ensure risk level matches combined score
        if risk_probability >= 70:
            risk_level = "high"
        elif risk_probability >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Determine bar style
        progress_width = f"{risk_probability}%"
        if risk_level == "low":
            progress_color = "bg-success"
        elif risk_level == "medium":
            progress_color = "bg-warning"
        else:
            progress_color = "bg-danger"

        print("Final Risk Level:", risk_level)
        print("Final Risk Percentage:", risk_probability)

        recommendation = "Stay hydrated and take breaks from the heat."
        if risk_level == "medium":
            recommendation = "Limit outdoor activities, stay hydrated, and check on vulnerable neighbors."
        elif risk_level == "high":
            recommendation = "Stay indoors in air-conditioned environments, drink plenty of water, and seek cool areas."

        alerts = {
            'government': f"Heatwave risk in {city}: {risk_level.upper()} - {explanation}",
            'ngo': f"Heatwave risk in {city}: {risk_level.upper()} - Prepare cooling stations and water distribution",
            'public': f"Heatwave risk in {city}: {risk_level.upper()} - {recommendation}"
        } if risk_probability >= 40 else None

        pdf_path = None
        if generate_pdf and risk_probability >= 40:
            data = {'max_temp': max_temp, 'humidity': humidity, 'consecutive_hot_days': 1}
            pdf_path = generate_alert_pdf('heatwave', risk_level, city, data)

        return render_template('result.html',
                              risk_detected=(risk_probability >= 40),
                              risk_probability=risk_probability,
                              risk_percentage=risk_probability,
                              risk_level=risk_level,
                              progress_width=progress_width,
                              progress_color=progress_color,
                              alerts=alerts,
                              severity=risk_level,
                              disaster_type='Heat Wave',
                              pdf_path=pdf_path,
                              city=city,
                              input_data={'max_temp': max_temp, 'min_temp': min_temp, 'humidity': humidity, 'wind_speed': wind_speed})

    return "Invalid disaster type", 400

# PDF download endpoint
@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    # Create required directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('generated_pdfs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)
