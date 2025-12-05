"""
Unified Breast Cancer Detection & Support Application
Combines mammogram detection with AI chatbot support
"""

from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging
import tempfile
from dotenv import load_dotenv
from flask_cors import CORS
from langdetect import detect_langs, DetectorFactory
from groq import Groq
from flask import Flask, render_template, request, send_file
from flask import request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage

import io
import base64
from flask_sqlalchemy import SQLAlchemy



# Ensure consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Handle imports
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efn_preprocess
except ImportError:
    from keras.applications.efficientnet import preprocess_input as efn_preprocess

app = Flask(__name__)
CORS(app)
# -------------------- DATABASE CONFIG --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/breastcare'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==================== CONFIGURATION ====================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}
MODEL_PATH = "EfficientNetB0_best.keras"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ==================== LOAD ULTRASOUND MODEL ====================
print("🔄 Loading ultrasound model...")
try:
    ultrasound_model = tf.keras.models.load_model(
        'static/model/breast_cancer_model.keras'
    )
    print("✅ Ultrasound model loaded successfully!")
except Exception as e:
    print("❌ Failed to load ultrasound model:", e)
    ultrasound_model = None


# ==================== LOAD MAMMOGRAM MODEL ====================
print("🔄 Loading mammogram model...")
try:
    mammo_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Mammogram model loaded successfully!")
except Exception as e:
    print("❌ Failed to load mammogram model:", e)
    mammo_model = None


# ==================== INITIALIZE GROQ CLIENT ====================
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    logger.warning("GROQ_API_KEY not found in .env file! Chatbot features will be disabled.")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=groq_api_key, timeout=120.0, max_retries=3)
        logger.info("✅ Groq client initialized - Chatbot features enabled")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None

# ==================== CHATBOT CONFIGURATION ====================
SYSTEM_PROMPTS = {
    "en": """You are a compassionate breast cancer support assistant. Your role is to:
1. Listen to patients' symptoms and concerns with empathy
2. Provide accurate information about breast cancer symptoms
3. Assess symptom severity and recommend appropriate action
4. Always encourage professional medical consultation for concerning symptoms
5. Offer emotional support and reassurance
IMPORTANT Guidelines:
- Be warm, clear, and supportive
- Never diagnose - help patients understand when to seek medical help
- Use phrases like "I understand your concern" and "You're not alone"
- Always respond in the same language as the patient's query
- For urgent symptoms (severe pain, bloody discharge, rapid changes), emphasize immediate medical attention
Symptom Severity Levels:
URGENT: Severe pain, bloody discharge, skin ulceration, rapid growth
HIGH: New lump, nipple retraction, persistent pain, swollen lymph nodes
MODERATE: Breast changes, nipple discharge, size/texture changes
LOW: Occasional tenderness, mild discomfort
Always end with encouragement and professional consultation reminder.""",

    "fr": """Vous êtes un assistant de soutien compatissant pour le cancer du sein. Votre rôle est de :
1. Écouter les symptômes et préoccupations des patientes avec empathie
2. Fournir des informations précises sur les symptômes du cancer du sein
3. Évaluer la gravité des symptômes et recommander des actions appropriées
4. Toujours encourager une consultation médicale professionnelle pour les symptômes préoccupants
5. Offrir un soutien émotionnel et des réconfortements
Directives IMPORTANTES :
- Soyez chaleureux, clair et solidaire
- Ne diagnostiquez jamais - aidez les patientes à comprendre quand consulter
- Utilisez des phrases comme "Je comprends votre inquiétude" et "Vous n'êtes pas seule"
- Répondez toujours dans la même langue que la patiente
- Pour les symptômes urgents (douleur sévère, écoulement sanglant, changements rapides), insistez sur une attention médicale immédiate
Niveaux de gravité :
URGENT : Douleur sévère, écoulement sanglant, ulcération cutanée, croissance rapide
ÉLEVÉ : Nouvelle bosse, rétraction du mamelon, douleur persistante, ganglions enflés
MODÉRÉ : Changements mammaires, écoulement du mamelon, changement taille/texture
FAIBLE : Sensibilité occasionnelle, inconfort léger
Terminez toujours avec des encouragements et un rappel de consultation professionnelle.""",

    "ar": """أنت مساعد دعم متعاطف لسرطان الثدي. دورك هو:
1. الاستماع لأعراض ومخاوف المرضى بتعاطف
2. تقديم معلومات دقيقة عن أعراض سرطان الثدي
3. تقييم خطورة الأعراض والتوصية بالإجراءات المناسبة
4. دائمًا تشجيع الاستشارة الطبية المهنية للأعراض المثيرة للقلق
5. تقديم الدعم العاطفي والطمأنينة
إرشادات مهمة:
- كن دافئًا وواضحًا وداعمًا
- لا تشخص أبدًا - ساعد المرضى على فهم متى يجب طلب المساعدة الطبية
- استخدم عبارات مثل "أفهم قلقك" و "لستِ وحدك"
- أجب دائمًا بنفس لغة المريض
- للأعراض العاجلة (ألم شديد، إفرازات دموية، تغيرات سريعة)، شدد على الرعاية الطبية الفورية
مستويات الخطورة:
عاجل: ألم شديد، إفرازات دموية، تقرح الجلد، نمو سريع
عالي: كتلة جديدة، انكماش الحلمة، ألم مستمر، تورم الغدد الليمفاوية
متوسط: تغيرات الثدي، إفرازات الحلمة، تغير الحجم/الملمس
منخفض: حساسية عرضية، انزعاج خفيف
أنهِ دائمًا بالتشجيع والتذكير باستشارة متخصص."""
}

SYMPTOM_KEYWORDS = {
    'urgent': ['bloody', 'blood', 'severe pain', 'rapid growth', 'ulcer', 'difficulty breathing'],
    'high': ['lump', 'mass', 'nipple retraction', 'dimpling', 'persistent pain', 'swollen lymph'],
    'moderate': ['discharge', 'nipple discharge', 'breast pain', 'size change', 'texture change', 'redness'],
    'low': ['tenderness', 'occasional pain', 'mild discomfort', 'soreness']
}


# ==================== MAMMOGRAM FUNCTIONS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def crop_mammo(img):
    """Simple auto-crop - exactly matches training preprocessing"""
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
    return img


def preprocess_image(image_path):
    """Preprocess image EXACTLY like training data"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image")

        img = crop_mammo(img)
        img = cv2.resize(img, (512, 512))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_array = img_rgb.astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efn_preprocess(img_array)

        return img_array
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")


def predict_image(image_path):
    """Make prediction on preprocessed image"""
    if mammo_model is None:
        return {"error": "Model not loaded"}

    try:
        img_array = preprocess_image(image_path)
        prediction = mammo_model.predict(img_array, verbose=0)[0][0]

        probability = float(prediction)
        is_malignant = probability > 0.5
        confidence = probability if is_malignant else (1 - probability)

        result = {
            "prediction": "MALIGNANT" if is_malignant else "BENIGN",
            "probability_malignant": round(probability * 100, 2),
            "probability_benign": round((1 - probability) * 100, 2),
            "confidence": round(confidence * 100, 2),
            "model_info": {
                "name": "EfficientNetB0",
                "auc": 0.8052,
                "accuracy": 70.14
            }
        }
        return result
    except Exception as e:
        return {"error": str(e)}


# ==================== CHATBOT FUNCTIONS ====================
def detect_language(text):
    """Detect language from text"""
    try:
        langs = detect_langs(text)
        supported_langs = ['en', 'fr', 'ar']
        for lang_prob in langs:
            lang = lang_prob.lang
            if lang in supported_langs:
                logger.info(f"Detected language: {lang} (confidence: {lang_prob.prob:.2f})")
                return lang
        logger.info("No supported language detected, defaulting to 'en'")
        return 'en'
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return "en"


def analyze_symptoms_simple(text):
    """Simple keyword-based symptom analysis"""
    text_lower = text.lower()
    detected_symptoms = []
    severity = "low"

    for keyword in SYMPTOM_KEYWORDS['urgent']:
        if keyword in text_lower:
            detected_symptoms.append(keyword)
            severity = "urgent"
            break
    if severity == "low":
        for keyword in SYMPTOM_KEYWORDS['high']:
            if keyword in text_lower:
                detected_symptoms.append(keyword)
                severity = "high"
                break
    if severity == "low":
        for keyword in SYMPTOM_KEYWORDS['moderate']:
            if keyword in text_lower:
                detected_symptoms.append(keyword)
                severity = "moderate"
                break
    if severity == "low":
        for keyword in SYMPTOM_KEYWORDS['low']:
            if keyword in text_lower:
                detected_symptoms.append(keyword)
                break

    return {
        'symptoms': list(set(detected_symptoms)),
        'severity': severity,
        'confidence': 0.75 if detected_symptoms else 0.5
    }


def transcribe_audio_groq(audio_file_path, language=None):
    """Transcribe audio using Groq Whisper"""
    if not groq_client:
        return {'text': None, 'language': None, 'success': False, 'error': 'Groq client not initialized'}

    try:
        with open(audio_file_path, "rb") as file:
            params = {
                "file": file,
                "model": "whisper-large-v3",
                "response_format": "verbose_json",
            }
            if language:
                params["language"] = language
            transcription = groq_client.audio.transcriptions.create(**params)
            text = transcription.text
            detected_lang = transcription.language if hasattr(transcription, 'language') else None
            logger.info(f"Transcription successful: '{text[:50]}...'")
            return {'text': text, 'language': detected_lang, 'success': True}
    except Exception as e:
        logger.error(f"Groq Whisper transcription failed: {e}")
        return {'text': None, 'language': None, 'success': False, 'error': str(e)}


def process_audio_data(audio_data):
    """Process base64 audio data"""
    try:
        if ',' in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(audio_data)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        logger.info(f"Audio saved to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return None


def get_chat_response(user_input, language):
    """Generate chat response using Groq"""
    if not groq_client:
        return {
            'response': "I'm sorry, the chatbot service is currently unavailable.",
            'assessment': {
                'symptoms': [],
                'severity': 'LOW',
                'confidence': 0.0,
                'recommendations': ['Please consult a healthcare professional.'],
                'explanation': ''
            },
            'language': language
        }

    try:
        analysis = analyze_symptoms_simple(user_input)

        symptom_context = f"""Basic Symptom Analysis:
- Keywords detected: {', '.join(analysis['symptoms']) if analysis['symptoms'] else 'None'}
- Estimated severity: {analysis['severity']}
- Analysis confidence: {analysis['confidence']:.0%}
Use this to guide your empathetic response and recommendations."""

        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['en'])

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt + "\n\n" + symptom_context},
                {"role": "user", "content": user_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=800
        )
        response_text = chat_completion.choices[0].message.content

        severity_map = {
            'urgent': 'URGENT',
            'high': 'HIGH',
            'moderate': 'MODERATE',
            'low': 'LOW'
        }
        severity_label = severity_map.get(analysis['severity'], 'LOW')

        recommendations = {
            'urgent': ["Go to the emergency room immediately.", "Call your doctor now."],
            'high': ["See a doctor within 24-48 hours.", "Schedule a mammogram."],
            'moderate': ["Monitor symptoms and consult a doctor soon.", "Perform regular self-exams."],
            'low': ["Continue monitoring.", "Consult a healthcare provider for reassurance."]
        }.get(analysis['severity'], ["Always consult a healthcare professional."])

        assessment = {
            'symptoms': analysis['symptoms'],
            'severity': severity_label,
            'confidence': analysis['confidence'],
            'recommendations': recommendations,
            'explanation': ''
        }

        return {
            'response': response_text,
            'assessment': assessment,
            'language': language
        }
    except Exception as e:
        logger.error(f"Chat response generation failed: {e}", exc_info=True)
        return {
            'response': "I'm sorry, an error occurred. Please try again or consult a doctor.",
            'assessment': {
                'symptoms': [],
                'severity': 'LOW',
                'confidence': 0.0,
                'recommendations': ['Consult a doctor.'],
                'explanation': ''
            },
            'language': language
        }


# ==================== ROUTES - YOUR ORIGINAL PAGES ====================
@app.route('/')
def home():
    """Landing page"""
    return render_template('home.html')


@app.route('/mammogram')
def mammogram_page():
    """Mammogram detection page"""
    return render_template('mamogram.html')


@app.route('/ultrasound')
def ultrasound_page():
    """Ultrasound page"""
    return render_template('ultrasound.html')

# Patient form page
@app.route("/form")
def form():
    return render_template("patient-form.html")

# Choose diagnosis page
@app.route("/choose-diagnosis")
def choose_diagnosis():
    return render_template("choose-diagnosis.html")

@app.route("/report")
def report():
    return render_template("report.html")

@app.post("/generate-report")
def generate_report():
    data = request.get_json()

    from reportlab.platypus import Image as RLImage  # IMPORTANT FIX

    patient = data.get("patient", {})
    mammogram = data.get("mammogram", {})
    mammogram_img_b64 = data["images"].get("mammogram")

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        name="Title",
        fontSize=22,
        leading=26,
        textColor=colors.black,
        alignment=1,
        spaceAfter=20,
    )

    section_header = ParagraphStyle(
        name="SectionHeader",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=10,
        spaceBefore=10,
    )

    normal = ParagraphStyle(
        name="Normal",
        fontSize=11,
        leading=16,
        textColor=colors.black,
    )

    story = []

    story.append(Paragraph("BreastCare AI – Diagnostic Report", title))
    story.append(Spacer(1, 12))

    # ------------------------- PATIENT -------------------------
    story.append(Paragraph("Patient Information", section_header))

    patient_table_data = [
        ["Name:", patient.get("name", "-")],
        ["Age:", patient.get("age", "-")],
        ["Email:", patient.get("email", "-")],
    ]

    patient_table = Table(patient_table_data, colWidths=[80, 350])
    patient_table.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)

    # ------------------------- MAMMOGRAM -------------------------
    if mammogram:
        story.append(Spacer(1, 15))
        story.append(Paragraph("Mammogram Analysis", section_header))

        mammo_data = [
            ["Diagnosis:", mammogram.get("diagnosis", "-")],
            ["Malignant Probability:", f"{mammogram.get('malignant', '-') } %"],
            ["Benign Probability:", f"{mammogram.get('benign', '-') } %"],
            ["Confidence:", f"{mammogram.get('confidence', '-') } %"],
        ]

        mammo_table = Table(mammo_data, colWidths=[160, 270])
        mammo_table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(mammo_table)

    if mammogram_img_b64:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Uploaded Mammogram Image", section_header))

        decoded_img = base64.b64decode(mammogram_img_b64.split(",")[1])
        img_io = io.BytesIO(decoded_img)

        story.append(RLImage(img_io, width=4.5 * inch, height=4.5 * inch))  # FIXED

    # ------------------------- ULTRASOUND -------------------------
    ultrasound = data.get("ultrasound", {})
    ultrasound_img_b64 = data["images"].get("ultrasound")

    if ultrasound:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Ultrasound Analysis", section_header))

        us_table_data = [
            ["Diagnosis:", ultrasound.get("diagnosis", "-")],
            ["Malignant Probability:", f"{ultrasound.get('probability_malignant', '-')} %"],
            ["Benign Probability:", f"{ultrasound.get('probability_benign', '-')} %"],
            ["Confidence:", f"{ultrasound.get('confidence', '-')} %"],
        ]

        us_table = Table(us_table_data, colWidths=[160, 270])
        us_table.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 11)]))
        story.append(us_table)

    if ultrasound_img_b64:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Ultrasound Image", section_header))

        decoded_us = base64.b64decode(ultrasound_img_b64.split(",")[1])
        us_io = io.BytesIO(decoded_us)

        story.append(RLImage(us_io, width=4.5 * inch, height=4.5 * inch))  # FIXED

    # ------------------------- TREATMENT -------------------------
    treatment = data.get("treatment", {})

    if treatment:
        story.append(Spacer(1, 20))
        story.append(Paragraph("Treatment Recommendation", section_header))

        tr_data = [
            ["Recommended Treatment:", treatment.get("recommended", "-")],
            ["Confidence:", f"{treatment.get('confidence', '-')} %"],
            ["Alternative Options:", ", ".join(treatment.get("alternatives", []))],
        ]

        tr_table = Table(tr_data, colWidths=[160, 270])
        tr_table.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 11)]))
        story.append(tr_table)

    # ------------------------- EXPORT -------------------------
    doc.build(story)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="BreastCare_Report.pdf",
        mimetype="application/pdf"
    )
# ==================== ROUTES - MAMMOGRAM PREDICTION ====================
@app.route('/predict', methods=['POST'])
def predict():
    """Predict mammogram classification + save to database + decision engine"""

    # 1️⃣ GET PATIENT ID
    patient_id = request.args.get("patient_id")
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400

    # 2️⃣ VALIDATE FILE
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, DCM"}), 400

    try:
        # 3️⃣ SAVE TEMP FILE
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 4️⃣ RUN MODEL PREDICTION
        mammo_result = predict_image(filepath)

        if "error" in mammo_result:
            os.remove(filepath)
            return jsonify(mammo_result), 500

        # 5️⃣ ENCODE IMAGE
        with open(filepath, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode('utf-8')
        os.remove(filepath)

        # 6️⃣ SAVE TO DATABASE
        diagnosis_record = Diagnosis(
            patient_id=patient_id,
            type="mammogram",
            diagnosis=mammo_result["prediction"],
            malignant=mammo_result["probability_malignant"],
            benign=mammo_result["probability_benign"],
            confidence=mammo_result["confidence"],
            image_base64=encoded_img
        )
        db.session.add(diagnosis_record)
        db.session.commit()

        # 7️⃣ FETCH PATIENT (for decision engine)
        patient = Patient.query.get(patient_id)

        patient_data = {
            "name": patient.name,
            "age": patient.age,
            "email": patient.email
        }

        # 8️⃣ RUN CLINICAL DECISION ENGINE
        decision = clinical_decision_engine(patient_data, mammo_result)

        # 9️⃣ RETURN EVERYTHING
        return jsonify({
            "success": True,
            "diagnosis_id": diagnosis_record.id,
            "mammo_result": mammo_result,
            "decision": decision
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# CLINICAL DECISION ENGINE
# ============================================

def clinical_decision_engine(patient, mammo_result):
    """
    patient  -> dict (name, age, email)
    mammo_result -> prediction result from model
    """
    decision = {
        "need_ultrasound": False,
        "need_treatment_reco": False,
        "risk_level": None,
        "final_label": mammo_result["prediction"],
        "actions": []
    }

    diagnosis = mammo_result["prediction"]
    confidence = mammo_result["confidence"]

    # 🔥 High-risk case → send to ultrasound
    if diagnosis == "MALIGNANT" and confidence >= 80:
        decision["need_ultrasound"] = True
        decision["risk_level"] = "HIGH"
        decision["actions"].append(
            "Suspicious mammogram detected — please proceed with ultrasound imaging."
        )

    # ⚠ Medium case → optional ultrasound
    elif diagnosis == "MALIGNANT" and confidence < 80:
        decision["risk_level"] = "MEDIUM"
        decision["actions"].append(
            "Mammogram suggests malignancy but with moderate confidence — ultrasound recommended."
        )
        decision["need_ultrasound"] = True

    # ✔ Benign → no ultrasound needed
    else:
        decision["risk_level"] = "LOW"
        decision["actions"].append(
            "Mammogram appears benign — no further imaging required."
        )

    return decision


# ==================== ROUTES - CHATBOT API ====================
@app.route('/api/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """Transcribe voice to text"""
    logger.info("VOICE TRANSCRIPTION REQUEST")
    try:
        data = request.json
        audio_data = data.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        audio_path = process_audio_data(audio_data)
        if not audio_path:
            return jsonify({'error': 'Failed to process audio'}), 500

        result = transcribe_audio_groq(audio_path)
        try:
            os.unlink(audio_path)
        except:
            pass

        if not result['success']:
            return jsonify({'error': 'Transcription failed', 'details': result.get('error')}), 500

        text_lang = detect_language(result['text'])
        return jsonify({
            'transcription': result['text'],
            'language': text_lang,
            'audio_language': result['language'],
            'success': True
        })
    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/chat', methods=['POST'])
def voice_chat():
    """Voice chat with AI assistant"""
    logger.info("VOICE CHAT REQUEST RECEIVED")
    try:
        data = request.json
        audio_data = data.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        audio_path = process_audio_data(audio_data)
        if not audio_path:
            return jsonify({'error': 'Failed to process audio'}), 500

        transcription_result = transcribe_audio_groq(audio_path)
        try:
            os.unlink(audio_path)
        except:
            pass

        if not transcription_result['success']:
            return jsonify({'error': 'Transcription failed'}), 500

        transcription_text = transcription_result['text']
        language = detect_language(transcription_text)
        chat_result = get_chat_response(transcription_text, language)

        return jsonify({
            'transcription': transcription_text,
            'response': chat_result['response'],
            'assessment': chat_result['assessment'],
            'language': language,
            'success': True
        })
    except Exception as e:
        logger.error(f"Voice chat error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/text/chat', methods=['POST'])
def text_chat():
    """Text chat with AI assistant"""
    logger.info("TEXT CHAT REQUEST RECEIVED")
    try:
        data = request.json
        user_message = data.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        language = detect_language(user_message)
        chat_result = get_chat_response(user_message, language)
        return jsonify(chat_result)
    except Exception as e:
        logger.error(f"Text chat error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test chatbot availability"""
    return jsonify({
        'status': 'ok',
        'whisper': 'available' if groq_client else 'unavailable',
        'chatbot': 'enabled' if groq_client else 'disabled',
        'supported_languages': ['en', 'fr', 'ar'],
        'audio_formats': ['webm', 'mp3', 'wav', 'ogg']
    })


# ==================== ROUTES - HEALTH & INFO ====================
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mammogram_model_loaded": mammo_model is not None,
        "chatbot_enabled": groq_client is not None,
        "model_name": "EfficientNetB0",
        "model_performance": {
            "auc": 0.8052,
            "accuracy": 70.14
        }
    })


@app.route('/model-info')
def model_info():
    """Get detailed model information"""
    if mammo_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify({
        "model_name": "EfficientNetB0",
        "input_shape": str(mammo_model.input_shape),
        "output_shape": str(mammo_model.output_shape),
        "performance": {
            "accuracy": 70.14,
            "auc_roc": 0.8052,
            "sensitivity": 85.0,
            "specificity": 60.0
        },
        "training_info": {
            "dataset": "CBIS-DDSM",
            "image_size": "512x512",
            "preprocessing": "Auto-crop + EfficientNet normalization"
        }
    })
# ======================================================
# DATABASE MODELS
# ======================================================

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    email = db.Column(db.String(120), unique=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())


class Diagnosis(db.Model):
    __tablename__ = 'diagnoses'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'))
    type = db.Column(db.String(50))
    diagnosis = db.Column(db.String(50))
    malignant = db.Column(db.Float)
    benign = db.Column(db.Float)
    confidence = db.Column(db.Float)
    image_base64 = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    patient = db.relationship("Patient", backref=db.backref("diagnoses", lazy=True))

class Admin(db.Model):
        __tablename__ = 'admins'
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(100), unique=True)
        password_hash = db.Column(db.String(200))


@app.post("/save-patient")
def save_patient():
    data = request.get_json()

    # 🔍 Check if user exists
    existing = Patient.query.filter_by(email=data["email"]).first()
    if existing:
        return jsonify({"patient_id": existing.id})

    # 👤 Create new patient
    patient = Patient(
        name=data["name"],
        age=data["age"],
        email=data["email"]
    )

    db.session.add(patient)
    db.session.commit()

    return jsonify({"patient_id": patient.id})

@app.route("/api/admin/patients")
def admin_patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()

    output = []
    for p in patients:
        diag_count = Diagnosis.query.filter_by(patient_id=p.id).count()
        last_diag = Diagnosis.query.filter_by(patient_id=p.id).order_by(Diagnosis.created_at.desc()).first()

        output.append({
            "id": p.id,
            "name": p.name,
            "age": p.age,
            "email": p.email,
            "created_at": str(p.created_at),
            "diagnosis_count": diag_count,
            "last_diagnosis": str(last_diag.created_at) if last_diag else "No records"
        })

    return jsonify(output)
@app.route("/api/history/<int:patient_id>")
def get_history(patient_id):
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    diagnoses = Diagnosis.query.filter_by(patient_id=patient_id).order_by(Diagnosis.created_at.desc()).all()

    data = []
    for d in diagnoses:
        data.append({
            "id": d.id,
            "type": d.type,
            "diagnosis": d.diagnosis,
            "malignant": d.malignant,
            "benign": d.benign,
            "confidence": d.confidence,
            "created_at": str(d.created_at),
            "image_base64": d.image_base64
        })

    return jsonify({
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "email": patient.email
        },
        "history": data
    })
@app.route("/history")
def history_page():
    return render_template("history.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")
from werkzeug.security import check_password_hash

@app.route("/admin-login", methods=["GET"])
def admin_login_page():
    return render_template("admin-login.html")

@app.post("/admin-login")
def admin_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    admin = Admin.query.filter_by(username=username).first()

    if not admin or not check_password_hash(admin.password_hash, password):
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    return jsonify({"success": True})

@app.route("/api/admin/diagnoses")
def admin_diagnoses():
    diagnoses = Diagnosis.query.order_by(Diagnosis.created_at.desc()).all()

    output = []
    for d in diagnoses:
        output.append({
            "id": d.id,
            "patient_id": d.patient_id,
            "diagnosis": d.diagnosis,
            "type": d.type,
            "malignant": d.malignant,
            "benign": d.benign,
            "confidence": d.confidence,
            "created_at": str(d.created_at)
        })

    return jsonify(output)


# Add these routes to your Flask app.py

# ==================== ROUTES FOR PATIENT HISTORY ====================
@app.route('/api/patients/<int:patient_id>')
def get_patient_details(patient_id):
    """Get detailed patient information"""
    try:
        patient = Patient.query.get(patient_id)
        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        return jsonify({
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "email": patient.email,
            "gender": "Female",  # You should add a gender field to your Patient model
            "created_at": str(patient.created_at)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/patients/<int:patient_id>/diagnoses')
def get_patient_diagnoses(patient_id):
    """Get diagnoses for a specific patient"""
    try:
        diagnoses = Diagnosis.query.filter_by(patient_id=patient_id).order_by(Diagnosis.created_at.desc()).all()

        result = []
        for d in diagnoses:
            result.append({
                "id": d.id,
                "type": d.type,
                "diagnosis": d.diagnosis,
                "malignant": d.malignant,
                "benign": d.benign,
                "normal": 100 - (d.malignant or 0) - (d.benign or 0) if d.type == "ultrasound" else None,
                "confidence": d.confidence,
                "created_at": str(d.created_at),
                "image_base64": d.image_base64
            })

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Add this to your existing routes section
@app.route('/predict-ultrasound', methods=['POST'])
def predict_ultrasound():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 1. Ouvre + resize intelligent 512x512
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 2. Prédiction
        pred = ultrasound_model.predict(img_array, verbose=0)[0]

        # 3. Applique softmax si nécessaire
        probabilities = tf.nn.softmax(pred).numpy() if len(pred) > 1 else [1.0 - pred[0], pred[0]]

        # 4. DÉTECTION AUTOMATIQUE DU NOMBRE DE CLASSES
        num_classes = len(probabilities)

        if num_classes == 2:
            # Cas le plus courant : que Benign et Malignant
            prob_benign = float(probabilities[0])
            prob_malignant = float(probabilities[1])
            prob_normal = 0.0
            classes = ['Benign', 'Malignant']
        elif num_classes == 3:
            prob_benign = float(probabilities[0])
            prob_malignant = float(probabilities[1])
            prob_normal = float(probabilities[2])
            classes = ['Benign', 'Malignant', 'Normal']
        else:
            # Sécurité
            prob_benign = prob_malignant = prob_normal = 0.0
            classes = ['Unknown']

        # Diagnostic final
        idx = np.argmax(probabilities)
        diagnosis = classes[idx] if num_classes >= 2 else "Unknown"
        confidence = float(probabilities[idx]) * 100

        result = {
            "diagnosis": diagnosis,
            "confidence": round(confidence, 1),
            "probability_benign": round(prob_benign * 100, 1),
            "probability_malignant": round(prob_malignant * 100, 1),
            "probability_normal": round(prob_normal * 100, 1)
        }

        print("Prédiction ultrasound RÉUSSIE :", result)
        # ============================
        # SAVE ULTRASOUND TO DATABASE
        # ============================
        patient_id = request.args.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400

        # Convert uploaded ultrasound image to Base64
        file.stream.seek(0)
        encoded_img = base64.b64encode(file.stream.read()).decode('utf-8')

        diagnosis_record = Diagnosis(
            patient_id=patient_id,
            type="ultrasound",
            diagnosis=diagnosis,
            malignant=prob_malignant * 100,
            benign=prob_benign * 100,
            confidence=confidence,
            image_base64=encoded_img
        )

        db.session.add(diagnosis_record)
        db.session.commit()

        result["diagnosis_id"] = diagnosis_record.id  # return ID to frontend

        return jsonify(result)


    except Exception as e:
        print("Erreur critique ultrasound:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Analysis failed"}), 500
@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')
@app.route('/predict-treatment', methods=['POST'])
def predict_treatment():
    try:
        data = request.get_json()

        # Extraction des données (tu peux les utiliser pour un modèle ML plus tard)
        age = float(data.get('age_at_diagnosis', 0))
        tumor_size = float(data.get('tumor_size', 0))
        stage = data.get('tumor_stage', '')
        lymph_nodes = int(data.get('lymph_nodes_positive', 0))
        npi = float(data.get('nottingham_index', 0))
        grade = data.get('grade', '')
        menopausal = data.get('menopausal', '')
        er = data.get('er', '')
        pr = data.get('pr', '')
        her2 = data.get('her2', '')

        # LOGIQUE DE RECOMMANDATION BASÉE SUR LES GUIDELINES (NCCN, ESMO, etc.)
        # Tu peux remplacer ça par un modèle ML plus tard

        recommendation = ""
        confidence = 0
        alternatives = []

        # 1. Cas clair : Triple négatif → Chimio
        if er == "negative" and pr == "negative" and her2 == "negative":
            recommendation = "Chemotherapy"
            confidence = 98
            alternatives = ["Neoadjuvant chemo", "Anthracycline + Taxane", "Immunotherapy if PD-L1+"]

        # 2. ER/PR+, HER2- → Hormonothérapie (+ chimio si haut risque)
        elif (er == "positive" or pr == "positive") and her2 == "negative":
            recommendation = "Hormone Therapy"
            confidence = 95
            if age < 50 or tumor_size > 50 or lymph_nodes > 3 or grade == "3":
                recommendation = "Hormone + Chemotherapy"
                confidence = 92
                alternatives = ["Tamoxifen", "Aromatase inhibitor + OFS", "Abemaciclib if high risk"]
            else:
                alternatives = ["Tamoxifen (pre-meno)", "Aromatase inhibitor (post-meno)", "Ovarian suppression"]

        # 3. HER2+ → Targeted therapy obligatoire
        elif her2 == "positive":
            recommendation = "Targeted Anti-HER2 Therapy"
            confidence = 99
            alternatives = ["Trastuzumab", "Pertuzumab + Trastuzumab", "T-DM1", "Neratinib if HR+"]
            if er == "positive" or pr == "positive":
                recommendation = "Anti-HER2 + Hormone Therapy"
            else:
                recommendation = "Anti-HER2 + Chemotherapy"

        # 4. Par défaut
        else:
            recommendation = "Multidisciplinary evaluation required"
            confidence = 85
            alternatives = ["Review case in tumor board"]
        # =====================================
        # SAVE TREATMENT RECOMMENDATION TO DB
        # =====================================
        patient_id = request.args.get("patient_id")
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400

        diagnosis_record = Diagnosis(
            patient_id=patient_id,
            type="treatment",
            diagnosis=recommendation,
            malignant=None,
            benign=None,
            confidence=confidence,
            image_base64=None
        )

        db.session.add(diagnosis_record)
        db.session.commit()

        # include DB ID in response
        diagnosis_id = diagnosis_record.id

        return jsonify({
            "recommended": recommendation,
            "confidence": confidence,
            "alternatives": alternatives,
            "diagnosis_id": diagnosis_id
        })


    except Exception as e:
        print("Erreur predict-treatment:", e)
        return jsonify({"error": "Invalid data"}), 400
@app.route("/api/report/<int:patient_id>")
def api_report(patient_id):
    """Return all data needed for the unified diagnostic report"""

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    # Fetch LAST mammogram
    mammogram = Diagnosis.query.filter_by(
        patient_id=patient_id, type="mammogram"
    ).order_by(Diagnosis.created_at.desc()).first()

    # Fetch LAST ultrasound
    ultrasound = Diagnosis.query.filter_by(
        patient_id=patient_id, type="ultrasound"
    ).order_by(Diagnosis.created_at.desc()).first()

    # Fetch LAST treatment recommendation
    treatment = Diagnosis.query.filter_by(
        patient_id=patient_id, type="treatment"
    ).order_by(Diagnosis.created_at.desc()).first()

    return jsonify({
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "email": patient.email,
            "created_at": str(patient.created_at)
        },
        "mammogram": {
            "diagnosis": mammogram.diagnosis if mammogram else None,
            "malignant": mammogram.malignant if mammogram else None,
            "benign": mammogram.benign if mammogram else None,
            "confidence": mammogram.confidence if mammogram else None,
            "image_base64": mammogram.image_base64 if mammogram else None,
        },
        "ultrasound": {
            "diagnosis": ultrasound.diagnosis if ultrasound else None,
            "malignant": ultrasound.malignant if ultrasound else None,
            "benign": ultrasound.benign if ultrasound else None,
            "confidence": ultrasound.confidence if ultrasound else None,
            "image_base64": ultrasound.image_base64 if ultrasound else None,
        },
        "treatment": {
            "recommended": treatment.diagnosis if treatment else None,
            "confidence": treatment.confidence if treatment else None,
            "alternatives": ["—"],
        }
    })

# ==================== RUN SERVER ====================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 UNIFIED BREAST CANCER DETECTION & SUPPORT APPLICATION")
    print("=" * 70)
    print(f"📊 Mammogram Model: {'✅ Loaded' if mammo_model else '❌ Not Loaded'}")
    print(f"   └─ EfficientNetB0 (AUC: 0.8052, Accuracy: 70.14%)")
    print(f"💬 Chatbot: {'✅ Enabled' if groq_client else '❌ Disabled'}")
    print(f"   └─ Groq Whisper + LLaMA 3.3 70B")
    print(f"   └─ Languages: English, French, Arabic")
    print(f"\n🌐 Server running at: http://localhost:5000")
    print(f"📁 Endpoints:")
    print(f"   ├─ /                    → Home page")
    print(f"   ├─ /mammogram           → Mammogram detection")
    print(f"   ├─ /ultrasound          → Ultrasound page")
    print(f"   ├─ /predict             → Mammogram prediction API")
    print(f"   ├─ /api/text/chat       → Text chatbot")
    print(f"   ├─ /api/voice/chat      → Voice chatbot")
    print(f"   ├─ /health              → Health check")
    print(f"   └─ /model-info          → Model information")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)