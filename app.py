from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import random
import sqlite3
from datetime import datetime
import logging
from typing import Union, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize database
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    
    # Create patients table
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  national_id TEXT,
                  nationality TEXT,
                  age INTEGER,
                  mobile_number TEXT,
                  gender TEXT,
                  chronic_diseases TEXT,
                  cholesterol REAL,
                  blood_pressure TEXT,
                  weight REAL,
                  height REAL,
                  smoking_status TEXT,
                  exercise_frequency TEXT,
                  probability REAL,
                  diagnosis TEXT,
                  disease_type TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Create treatments table
    c.execute('''CREATE TABLE IF NOT EXISTS treatments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  treatment_type TEXT,
                  description TEXT,
                  start_date DATETIME,
                  end_date DATETIME,
                  FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    
    conn.commit()
    conn.close()

init_db()

# Load the trained heart disease models
model2_path = "camus_rnn_model.h5"
model3_path = "camus_segmentation_classification_model.h5"

try:
    model2 = load_model(model2_path, compile=False)
    if hasattr(model2.layers[0], 'time_major'):
        model2.layers[0].time_major = False
    
    model3 = load_model(model3_path, compile=False)
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.info("Creating dummy models for development purposes...")
    model2 = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(2, activation='softmax')
    ])
    model3 = keras.Sequential([
        keras.layers.InputLayer(input_shape=(224, 224, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(2, activation='softmax')
    ])

# Initialize model1 with the same architecture
model1 = keras.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation='softmax')
])

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    if image is None:
        raise ValueError("Image not found. Please check the path.")
    
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    
    if image.shape[-1] != 3:
        image = image[..., :3]
    
    image = np.expand_dims(image, axis=0)
    return image

# Function to generate visualizations
def generate_visualizations(prediction1, prediction2, prediction3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    ax1.bar(['Normal', 'Heart Disease'], prediction1.flatten(), color=['green', 'red'])
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Probability')
    ax1.set_title('Model 1: Heart Disease Probability')
    
    ax2.bar(['Normal', 'Heart Disease'], prediction2.flatten(), color=['green', 'red'])
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Probability')
    ax2.set_title('Model 2: Heart Disease Probability')
    
    ax3.bar(['Normal', 'Heart Disease'], prediction3.flatten(), color=['green', 'red'])
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Probability')
    ax3.set_title('Model 3: Heart Disease Probability')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# Function to display the original image with colors
def display_colored_image(image_path, prediction1, prediction2, prediction3):
    image = Image.open(image_path)
    image = np.array(image) / 255.0
    
    avg_prob = (prediction1[0][1] + prediction2[0][1] + prediction3[0][1]) / 3
    final_prediction = "Heart Disease" if avg_prob > 0.5 else "Normal"
    
    plt.figure(figsize=(8, 8))
    plt.title(f"Diagnosis: {final_prediction}", fontsize=16, color='red' if final_prediction == "Heart Disease" else 'green')
    plt.imshow(image)
    plt.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Function to determine heart disease type with detailed information
def determine_heart_disease_type(patient_data):
    heart_disease_types = {
        "Coronary Artery Disease (CAD)": {
            "type": "Coronary Artery Disease (CAD)",
            "description": "Blockage in the coronary arteries that supply blood to the heart muscle.",
            "causes": [
                "Build-up of plaque in arteries (atherosclerosis)",
                "High cholesterol levels",
                "High blood pressure",
                "Smoking",
                "Diabetes",
                "Sedentary lifestyle"
            ],
            "symptoms": [
                "Chest pain (angina)",
                "Shortness of breath",
                "Heart attack",
                "Fatigue",
                "Irregular heartbeat"
            ],
            "treatment": [
                "Medications (statins, beta-blockers, aspirin)",
                "Angioplasty and stent placement",
                "Coronary artery bypass surgery",
                "Lifestyle changes (diet, exercise)"
            ],
            "prevention": [
                "Quit smoking",
                "Control blood pressure",
                "Manage diabetes",
                "Regular exercise",
                "Healthy diet low in saturated fats"
            ]
        },
        "Arrhythmia": {
            "type": "Arrhythmia",
            "description": "Abnormal heart rhythm that can be too fast, too slow, or irregular.",
            "causes": [
                "Heart attack or damaged heart tissue",
                "High blood pressure",
                "Diabetes",
                "Smoking",
                "Excessive alcohol or caffeine",
                "Drug abuse",
                "Stress"
            ],
            "symptoms": [
                "Fluttering in chest",
                "Racing or slow heartbeat",
                "Chest pain",
                "Shortness of breath",
                "Dizziness or fainting"
            ],
            "treatment": [
                "Medications (anti-arrhythmics)",
                "Pacemaker implantation",
                "Implantable cardioverter-defibrillator (ICD)",
                "Ablation therapy",
                "Lifestyle changes"
            ],
            "prevention": [
                "Limit alcohol and caffeine",
                "Quit smoking",
                "Manage stress",
                "Maintain healthy weight",
                "Control blood pressure"
            ]
        }
    }
    
    # Determine most likely type based on patient data
    if float(patient_data['cholesterol']) > 200 and patient_data['smoking_status'] == "Yes":
        selected_type = "Coronary Artery Disease (CAD)"
    elif patient_data['chronic_diseases'] in ["Hypertension", "High blood pressure"]:
        selected_type = "Arrhythmia"
    else:
        selected_type = random.choice(list(heart_disease_types.keys()))
    
    return heart_disease_types[selected_type]

# Function to save patient data to database
def save_patient_data(name, national_id, nationality, age, mobile_number, gender, 
                     chronic_diseases, cholesterol, blood_pressure, weight, height, 
                     smoking_status, exercise_frequency, probability, diagnosis, disease_type):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO patients 
                 (name, national_id, nationality, age, mobile_number, gender, 
                  chronic_diseases, cholesterol, blood_pressure, weight, height, 
                  smoking_status, exercise_frequency, probability, diagnosis, disease_type)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (name, national_id, nationality, age, mobile_number, gender, 
               chronic_diseases, cholesterol, blood_pressure, weight, height, 
               smoking_status, exercise_frequency, probability, diagnosis, disease_type))
    
    patient_id = c.lastrowid
    
    # If probability > 0.3, add to treatment monitoring
    if probability > 0.3:
        treatment_type = "Monitoring" if probability < 0.5 else "Active Treatment"
        
        c.execute('''INSERT INTO treatments 
                     (patient_id, treatment_type, description, start_date)
                     VALUES (?, ?, ?, ?)''',
                  (patient_id, treatment_type, 
                   f"Initial monitoring for potential heart disease (probability: {probability:.2f})", 
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    conn.commit()
    conn.close()
    return patient_id

# Function to predict and generate report
def predict_and_generate_report(image_path, name, national_id, nationality, age, mobile_number, gender, 
                              chronic_diseases, cholesterol, blood_pressure, weight, height, 
                              smoking_status, exercise_frequency):
    img_array = preprocess_image(image_path)
    
    try:
        prediction1 = model1.predict(img_array)
        prediction2 = model2.predict(img_array)
        prediction3 = model3.predict(img_array)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error in processing image. Please try another image.", None, None, None
    
    avg_prob = (prediction1[0][1] + prediction2[0][1] + prediction3[0][1]) / 3
    final_prediction = "Heart Disease" if avg_prob > 0.5 else "Normal"

    heart_disease_info = None
    if final_prediction == "Heart Disease" or avg_prob > 0.3:
        patient_data = {
            'cholesterol': cholesterol,
            'smoking_status': smoking_status,
            'chronic_diseases': chronic_diseases,
            'age': age,
            'gender': gender
        }
        heart_disease_info = determine_heart_disease_type(patient_data)
    
    visualization_buf = generate_visualizations(prediction1, prediction2, prediction3)
    colored_image_buf = display_colored_image(image_path, prediction1, prediction2, prediction3)
    
    report = (f"Name: {name}\nNational ID: {national_id}\nNationality: {nationality}\nAge: {age}\n"
              f"Mobile Number: {mobile_number}\nGender: {gender}\nChronic Diseases: {chronic_diseases}\n"
              f"Cholesterol: {cholesterol}\nBlood Pressure: {blood_pressure}\n"
              f"Weight: {weight}\nHeight: {height}\nSmoking Status: {smoking_status}\n"
              f"Exercise Frequency: {exercise_frequency}\n\n")
    
    report += f"Model Prediction: {final_prediction}\n"
    report += f"Average Model Probability: {avg_prob:.2f}\n"
    
    if final_prediction == "Heart Disease" or avg_prob > 0.3:
        report += f"\n=== HEART DISEASE DETAILS ===\n"
        report += f"Type: {heart_disease_info['type']}\n"
        report += f"Description: {heart_disease_info['description']}\n\n"
        
        report += "Possible Causes:\n"
        for cause in heart_disease_info['causes']:
            report += f"- {cause}\n"
        
        report += "\nCommon Symptoms:\n"
        for symptom in heart_disease_info['symptoms']:
            report += f"- {symptom}\n"
        
        report += "\nRecommended Treatments:\n"
        for treatment in heart_disease_info['treatment']:
            report += f"- {treatment}\n"
        
        report += "\nPrevention Tips:\n"
        for prevention in heart_disease_info['prevention']:
            report += f"- {prevention}\n"
        
        # Add monitoring recommendation if probability > 0.3 but < 0.5
        if 0.3 <= avg_prob < 0.5:
            report += "\n=== MONITORING RECOMMENDATION ===\n"
            report += "Based on your results (probability between 0.3-0.5), we recommend:\n"
            report += "- Regular follow-up appointments with your cardiologist\n"
            report += "- Lifestyle modifications (diet, exercise, stress management)\n"
            report += "- Periodic cardiac testing to monitor any changes\n"
    else:
        report += "Heart Disease Type: None\n"
        report += "Your heart appears healthy based on our analysis. Maintain a healthy lifestyle to prevent future problems.\n"
    
    # Save patient data to database
    save_patient_data(name, national_id, nationality, age, mobile_number, gender, 
                     chronic_diseases, cholesterol, blood_pressure, weight, height, 
                     smoking_status, exercise_frequency, avg_prob, 
                     final_prediction, heart_disease_info['type'] if heart_disease_info else "None")
    
    return report, visualization_buf, colored_image_buf, heart_disease_info

@app.route('/')
def index():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('static'):
        os.makedirs('static')
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process() -> Union[Response, Tuple[str, int]]:
    try:
        if request.method == 'POST':
            # Create directories if they don't exist
            os.makedirs('uploads', exist_ok=True)
            os.makedirs('static', exist_ok=True)
            
            # Get form data
            try:
                name = str(request.form['name'])
                national_id = str(request.form['national_id'])
                nationality = str(request.form['nationality'])
                age = int(request.form['age'])
                mobile_number = str(request.form['mobile_number'])
                gender = str(request.form['gender'])
                chronic_diseases = str(request.form['chronic_diseases'])
                cholesterol = float(request.form['cholesterol'])
                blood_pressure = str(request.form['blood_pressure'])
                weight = float(request.form['weight'])
                height = float(request.form['height'])
                smoking_status = str(request.form['smoking_status'])
                exercise_frequency = str(request.form['exercise_frequency'])
            except KeyError as e:
                logger.error(f"Missing form field: {str(e)}")
                return f"Missing required field: {str(e)}", 400
            except ValueError as e:
                logger.error(f"Invalid form data: {str(e)}")
                return f"Invalid data format: {str(e)}", 400
            
            # Handle image upload
            if 'image' not in request.files:
                logger.error("No image field in request")
                return "No image uploaded", 400
            
            image = request.files['image']
            if image.filename == '':
                logger.error("No selected file")
                return "No image selected", 400
            
            try:
                filename = str(image.filename)
                image_path = os.path.join('uploads', filename)
                image.save(image_path)
                logger.debug(f"Image saved to {image_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                return f"Error saving image: {str(e)}", 500
            
            try:
                report, visualization_buf, colored_image_buf, heart_disease_info = predict_and_generate_report(
                    image_path, name, national_id, nationality, age, mobile_number, gender,
                    chronic_diseases, cholesterol, blood_pressure, weight, height,
                    smoking_status, exercise_frequency)
                
                if not all([report, visualization_buf, colored_image_buf]):
                    raise ValueError("Failed to generate report or visualizations")
                
                visualization_path = os.path.join('static', 'visualization.png')
                with open(visualization_path, 'wb') as f:
                    f.write(visualization_buf.getbuffer())
                
                colored_image_path = os.path.join('static', 'colored_image.png')
                with open(colored_image_path, 'wb') as f:
                    f.write(colored_image_buf.getbuffer())
                
                return render_template('result.html', 
                                    report=report, 
                                    visualization=visualization_path, 
                                    colored_image=colored_image_path,
                                    age=age,
                                    gender=gender,
                                    chronic_diseases=chronic_diseases,
                                    cholesterol=cholesterol,
                                    heart_disease_info=heart_disease_info)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return f"Error processing image: {str(e)}", 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"An unexpected error occurred: {str(e)}", 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        report = request.form['report']
        visualization_path = request.form['visualization']
        colored_image_path = request.form['colored_image']
        age = request.form['age']
        gender = request.form['gender']
        chronic_diseases = request.form['chronic_diseases']
        cholesterol = request.form['cholesterol']
        
        # Create PDF with UTF-8 encoding
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 24)  # Increased from 16
        pdf.cell(0, 15, 'Heart Disease Detection Report', 0, 1, 'C')  # Increased height from 10
        pdf.ln(15)  # Increased from 10
        
        # Parse report lines
        info_dict = {}
        lines = report.split('\n')
        for line in lines:
            if ':' in line and len(line.split(':', 1)) == 2:
                key, value = line.split(':', 1)
                info_dict[key.strip()] = value.strip()
        
        # Patient Information Section
        pdf.set_font('Arial', 'B', 20)  # Increased from 14
        pdf.cell(0, 12, 'Patient Information', 0, 1, 'L')  # Increased height from 10
        pdf.ln(8)  # Increased from 5
        
        # Table Header
        pdf.set_fill_color(200, 200, 200)
        pdf.set_font('Arial', 'B', 14)
        # Draw header with darker background
        pdf.set_fill_color(180, 180, 180)
        pdf.cell(60, 12, 'Field', 1, 0, 'C', True)
        pdf.cell(120, 12, 'Value', 1, 1, 'C', True)
        pdf.set_fill_color(224, 235, 255)  # Reset fill color for alternating rows
        
        # Table Content
        data = [
            ['Name', info_dict.get('Name', 'N/A')],
            ['National ID', info_dict.get('National ID', 'N/A')],
            ['Nationality', info_dict.get('Nationality', 'N/A')],
            ['Age', info_dict.get('Age', 'N/A')],
            ['Mobile Number', info_dict.get('Mobile Number', 'N/A')],
            ['Gender', info_dict.get('Gender', 'N/A')],
            ['Chronic Diseases', info_dict.get('Chronic Diseases', 'N/A')],
            ['Cholesterol', info_dict.get('Cholesterol', 'N/A')],
            ['Blood Pressure', info_dict.get('Blood Pressure', 'N/A')],
            ['Weight', info_dict.get('Weight', 'N/A')],
            ['Height', info_dict.get('Height', 'N/A')],
            ['Smoking Status', info_dict.get('Smoking Status', 'N/A')],
            ['Exercise Frequency', info_dict.get('Exercise Frequency', 'N/A')]
        ]
        
        # Adjust cell heights based on content
        pdf.set_font('Arial', '', 12)  # Set font before calculating heights
        row_heights = []
        for row in data:
            # Calculate height needed for the value cell (second column)
            lines_needed = len(pdf.multi_cell(120, 10, str(row[1]), split_only=True))
            row_height = max(10, lines_needed * 10)  # Minimum 10, or more if needed
            row_heights.append(row_height)
        
        # Print table rows with alternating colors
        fill = False
        for i, row in enumerate(data):
            pdf.set_fill_color(224, 235, 255) if fill else pdf.set_fill_color(255, 255, 255)
            
            # Save current position
            x_pos = pdf.get_x()
            y_pos = pdf.get_y()
            
            # Print first column (label)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(60, row_heights[i], str(row[0]), 1, 0, 'L', 1)
            
            # Print second column (value)
            pdf.set_font('Arial', '', 12)
            pdf.set_xy(x_pos + 60, y_pos)
            pdf.multi_cell(120, row_heights[i], str(row[1]), 1, 'L', 1)
            
            # Move to next row
            pdf.set_xy(x_pos, y_pos + row_heights[i])
            
            fill = not fill
        
        pdf.ln(15)  # Increased from 10
        
        # Diagnosis Results Section
        pdf.set_font('Arial', 'B', 20)  # Increased from 14
        pdf.cell(0, 12, 'Diagnosis Results', 0, 1, 'L')  # Increased height from 10
        pdf.ln(8)  # Increased from 5
        
        # Get diagnosis information
        diagnosis_text = ""
        in_diagnosis = False
        for line in lines:
            if "Model Prediction:" in line:
                in_diagnosis = True
            if in_diagnosis:
                diagnosis_text += line + "\n"
        
        # Process diagnosis information
        pdf.set_font('Arial', '', 12)  # Increased from 10
        
        for line in diagnosis_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("==="):
                # Section header
                pdf.ln(8)  # Increased from 5
                pdf.set_font('Arial', 'B', 16)  # Increased from 12
                clean_text = line.replace("===", "").strip().encode('ascii', 'replace').decode('ascii')
                pdf.cell(0, 10, clean_text, 0, 1, 'L')  # Increased height from 8
                pdf.set_font('Arial', '', 12)  # Increased from 10
            elif ":" in line:
                # Key-value pair
                key, value = line.split(":", 1)
                clean_key = key.strip().encode('ascii', 'replace').decode('ascii')
                clean_value = value.strip().encode('ascii', 'replace').decode('ascii')
                
                # Special formatting for probability
                if "Probability" in clean_key:
                    pdf.ln(5)
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, clean_key + ":", 0, 1, 'L')
                    pdf.set_font('Arial', 'B', 20)
                    pdf.cell(0, 12, clean_value, 0, 1, 'C')
                    pdf.ln(5)
                else:
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(60, 8, clean_key + ":", 0, 0, 'L')
                    pdf.set_font('Arial', '', 12)
                    pdf.multi_cell(0, 8, clean_value, 0, 'L')
            elif line.startswith("-"):
                # Bullet point
                pdf.cell(15, 10, "", 0, 0)  # Increased indent and height
                pdf.cell(8, 10, "-", 0, 0)  # Increased spacing
                clean_text = line[1:].strip().encode('ascii', 'replace').decode('ascii')
                pdf.multi_cell(0, 10, clean_text, 0, 'L')  # Increased height from 8
            else:
                # Regular text
                clean_text = line.encode('ascii', 'replace').decode('ascii')
                pdf.multi_cell(0, 10, clean_text, 0, 'L')  # Increased height from 8
        
        # Add visualizations on new page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)  # Increased from 14
        pdf.cell(0, 12, 'Model Predictions', 0, 1, 'L')  # Increased height from 10
        pdf.image(visualization_path, x=10, y=pdf.get_y(), w=190)
        
        pdf.ln(10)
        pdf.cell(0, 12, 'Medical Image Analysis', 0, 1, 'L')  # Increased height from 10
        pdf.image(colored_image_path, x=10, y=pdf.get_y(), w=190)
        
        # Generate unique filename for each report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"heart_report_{timestamp}.pdf"
        
        # Ensure static directory exists
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        os.makedirs(static_dir, exist_ok=True)
        
        pdf_path = os.path.join(static_dir, pdf_filename)
        
        try:
            # Save PDF
            logger.info(f"Attempting to save PDF to: {pdf_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"Static directory path: {static_dir}")
            logger.info(f"PDF filename: {pdf_filename}")
            
            # Ensure visualization paths are absolute
            visualization_path = os.path.join(static_dir, os.path.basename(visualization_path))
            colored_image_path = os.path.join(static_dir, os.path.basename(colored_image_path))
            
            logger.info(f"Visualization path: {visualization_path}")
            logger.info(f"Colored image path: {colored_image_path}")
            
            # Verify images exist
            if not os.path.exists(visualization_path):
                raise FileNotFoundError(f"Visualization image not found at: {visualization_path}")
            if not os.path.exists(colored_image_path):
                raise FileNotFoundError(f"Colored image not found at: {colored_image_path}")
            
            pdf.output(pdf_path)
            logger.info("PDF generated successfully")
            
            # Delete old PDF files (keep only the last 5)
            pdf_files = sorted([f for f in os.listdir(static_dir) if f.startswith('heart_report_') and f.endswith('.pdf')])
            for old_pdf in pdf_files[:-5]:
                try:
                    os.remove(os.path.join(static_dir, old_pdf))
                except Exception as e:
                    logger.warning(f"Failed to delete old PDF {old_pdf}: {str(e)}")
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file was not created at {pdf_path}")
                
            return send_file(pdf_path, as_attachment=True)
        except Exception as e:
            logger.error(f"Failed to generate PDF report. Error details: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return f"An error occurred while generating the PDF: {str(e)}", 500

@app.route('/patient_list')
def patient_list():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    
    # Get patients with probability >= 0.3
    c.execute('''SELECT id, name, age, gender, CAST(probability AS FLOAT) as probability, 
                 diagnosis, disease_type, timestamp 
                 FROM patients 
                 WHERE CAST(probability AS FLOAT) >= 0.3 
                 ORDER BY probability DESC''')
    patients = c.fetchall()
    
    # Convert probability to float for each patient
    patients = [(id, name, age, gender, float(prob) if prob is not None else 0.0, 
                diagnosis, disease_type, timestamp) 
               for id, name, age, gender, prob, diagnosis, disease_type, timestamp in patients]
    
    conn.close()
    
    return render_template('patient_list.html', patients=patients)

@app.route('/patient_details/<int:patient_id>')
def patient_details(patient_id):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    
    # Get patient details with probability as float
    c.execute('''SELECT id, name, national_id, nationality, age, mobile_number, gender,
                 chronic_diseases, cholesterol, blood_pressure, weight, height,
                 smoking_status, exercise_frequency, CAST(probability AS FLOAT) as probability,
                 diagnosis, disease_type, timestamp
                 FROM patients WHERE id = ?''', (patient_id,))
    patient = c.fetchone()
    
    # Get treatments
    c.execute('''SELECT * FROM treatments WHERE patient_id = ?''', (patient_id,))
    treatments = c.fetchall()
    
    conn.close()
    
    if not patient:
        return "Patient not found", 404
    
    # Convert to dictionary for easier access in template
    patient_data = {
        'id': patient[0],
        'name': patient[1],
        'national_id': patient[2],
        'nationality': patient[3],
        'age': patient[4],
        'mobile_number': patient[5],
        'gender': patient[6],
        'chronic_diseases': patient[7],
        'cholesterol': patient[8],
        'blood_pressure': patient[9],
        'weight': patient[10],
        'height': patient[11],
        'smoking_status': patient[12],
        'exercise_frequency': patient[13],
        'probability': float(patient[14]) if patient[14] is not None else 0.0,
        'diagnosis': patient[15],
        'disease_type': patient[16],
        'timestamp': patient[17]
    }
    
    return render_template('patient_details.html', patient=patient_data, treatments=treatments)

if __name__ == '__main__':
    app.run(debug=True)