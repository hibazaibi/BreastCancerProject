from flask import Flask, request, jsonify, render_template, send_from_directory
from groq import Groq
import os
import logging
import base64
import tempfile
import json
from dotenv import load_dotenv
from flask_cors import CORS
from langdetect import detect_langs, DetectorFactory

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

app = Flask(__name__)
CORS(app)

# Initialize Groq client
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    timeout=120.0,
    max_retries=3
)

# Initialize symptom checker (import after to avoid circular import)
symptom_checker = None
try:
    from symptom_checker import VoiceSymptomChecker

    symptom_checker = VoiceSymptomChecker()
    logger.info("✅ Breast Cancer Symptom Checker initialized with Groq Whisper")
except ImportError as e:
    logger.warning(f"⚠️ Symptom checker not available: {e}")
    logger.info("ℹ️  Continuing without advanced symptom analysis")

# System prompts for different languages
SYSTEM_PROMPTS = {
    "en": """You are a compassionate breast cancer support assistant. Your role is to:
1. Listen to patients' symptoms and concerns with empathy
2. Provide accurate information about breast cancer symptoms
3. Assess symptom severity and recommend appropriate action
4. Always encourage professional medical consultation
5. Offer emotional support and reassurance

Be warm, clear, and supportive. Never diagnose, but help patients understand when to seek help.
Always respond in the same language as the patient's query.""",

    "fr": """Vous êtes un assistant de soutien compatissant pour le cancer du sein. Votre rôle est de :
1. Écouter les symptômes et préoccupations des patientes avec empathie
2. Fournir des informations précises sur les symptômes du cancer du sein
3. Évaluer la gravité des symptômes et recommander des actions appropriées
4. Toujours encourager une consultation médicale professionnelle
5. Offrir un soutien émotionnel et des réconfortements

Soyez chaleureux, clair et solidaire. Ne diagnostiquez jamais, mais aidez les patientes à comprendre quand consulter.
Répondez toujours dans la même langue que la patiente.""",

    "ar": """أنت مساعد دعم متعاطف لسرطان الثدي. دورك هو:
1. الاستماع لأعراض ومخاوف المرضى بتعاطف
2. تقديم معلومات دقيقة عن أعراض سرطان الثدي
3. تقييم خطورة الأعراض والتوصية بالإجراءات المناسبة
4. دائمًا تشجيع الاستشارة الطبية المهنية
5. تقديم الدعم العاطفي والطمأنينة

كن دافئًا وواضحًا وداعمًا. لا تشخص أبدًا، بل ساعد المرضى على فهم متى يجب طلب المساعدة.
أجب دائمًا بنفس لغة المريض."""
}


# -------------------- Language Detection --------------------
def detect_language(text):
    """Detect language using langdetect"""
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


# -------------------- Audio Transcription with Groq Whisper --------------------
def transcribe_audio_groq(audio_file_path, language=None):
    """Transcribe audio using Groq Whisper (supports WebM, MP3, WAV, etc.)"""
    try:
        with open(audio_file_path, "rb") as file:
            # Groq Whisper supports automatic language detection
            params = {
                "file": file,
                "model": "whisper-large-v3",
                "response_format": "verbose_json",  # Get language info
            }

            # Optionally specify language for better accuracy
            if language:
                params["language"] = language

            transcription = groq_client.audio.transcriptions.create(**params)

        # Extract text and detected language
        text = transcription.text
        detected_lang = transcription.language if hasattr(transcription, 'language') else None

        logger.info(f"✅ Transcription successful: '{text[:50]}...'")
        logger.info(f"✅ Detected audio language: {detected_lang}")

        return {
            'text': text,
            'language': detected_lang,
            'success': True
        }

    except Exception as e:
        logger.error(f"❌ Groq Whisper transcription failed: {e}")
        return {
            'text': None,
            'language': None,
            'success': False,
            'error': str(e)
        }


def process_audio_data(audio_data):
    """Process base64 audio data and save temporarily"""
    try:
        # Decode base64 audio
        if ',' in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(audio_data)

        # Save to temporary file (Groq supports WebM directly)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        logger.info(f"✅ Audio saved to: {temp_path}")
        return temp_path

    except Exception as e:
        logger.error(f"❌ Audio processing failed: {e}")
        return None


# -------------------- Chat Response with Symptom Analysis --------------------
def get_chat_response(user_input, language):
    """Generate chat response using Groq and symptom checker"""
    try:
        # First, analyze symptoms using the symptom checker (if available)
        assessment = None
        symptom_context = ""

        if symptom_checker:
            assessment = symptom_checker.process_text(user_input, language)

        # Build context for the LLM
        symptom_context = f"""
Symptom Assessment:
- Detected Symptoms: {', '.join(assessment.detected_symptoms) if assessment.detected_symptoms else 'None'}
- Severity Level: {assessment.severity.value}
- Confidence: {assessment.confidence:.2%}
- Recommendation Type: {assessment.recommendation_type.value}
"""

        # Get system prompt in detected language
        system_prompt = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS['en'])

        # Generate response using Groq
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

        return {
            'response': response_text,
            'assessment': {
                'symptoms': assessment.detected_symptoms,
                'severity': assessment.severity.value,
                'confidence': assessment.confidence,
                'recommendations': assessment.recommendations,
                'explanation': assessment.raw_response
            },
            'language': language
        }

    except Exception as e:
        logger.error(f"❌ Chat response generation failed: {e}")
        error_messages = {
            'en': "I'm sorry, I encountered an error. Please try again or seek medical advice directly.",
            'fr': "Désolé, j'ai rencontré une erreur. Veuillez réessayer ou consulter directement un médecin.",
            'ar': "عذرًا، واجهت خطأ. يرجى المحاولة مرة أخرى أو استشارة طبيب مباشرة."
        }
        return {
            'response': error_messages.get(language, error_messages['en']),
            'assessment': None,
            'language': language,
            'error': str(e)
        }


# -------------------- API Routes --------------------

@app.route('/')
def home():
    """Serve main HTML page"""
    return render_template('mamogram.html')


@app.route('/api/voice/transcribe', methods=['POST'])
def transcribe_voice():
    """Transcribe voice message and return text with detected language"""
    try:
        data = request.json
        audio_data = data.get('audio')

        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Process audio
        audio_path = process_audio_data(audio_data)
        if not audio_path:
            return jsonify({'error': 'Failed to process audio'}), 500

        # Transcribe with Groq Whisper
        transcription_result = transcribe_audio_groq(audio_path)

        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass

        if not transcription_result['success']:
            return jsonify({
                'error': 'Transcription failed',
                'details': transcription_result.get('error')
            }), 500

        # Map Whisper language codes to our supported languages
        whisper_lang = transcription_result['language']
        detected_lang = 'en'  # default
        if whisper_lang:
            if whisper_lang.startswith('fr'):
                detected_lang = 'fr'
            elif whisper_lang.startswith('ar'):
                detected_lang = 'ar'
            else:
                detected_lang = 'en'

        # Double-check with text-based detection
        text_lang = detect_language(transcription_result['text'])

        return jsonify({
            'transcription': transcription_result['text'],
            'language': text_lang or detected_lang,
            'audio_language': whisper_lang,
            'success': True
        })

    except Exception as e:
        logger.error(f"❌ Voice transcription error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/chat', methods=['POST'])
def voice_chat():
    """Complete voice chat: transcribe + analyze + respond"""
    try:
        data = request.json
        audio_data = data.get('audio')

        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Process audio
        audio_path = process_audio_data(audio_data)
        if not audio_path:
            return jsonify({'error': 'Failed to process audio'}), 500

        # Transcribe
        transcription_result = transcribe_audio_groq(audio_path)

        # Clean up
        try:
            os.unlink(audio_path)
        except:
            pass

        if not transcription_result['success']:
            return jsonify({'error': 'Transcription failed'}), 500

        transcription_text = transcription_result['text']

        # Detect language
        language = detect_language(transcription_text)

        # Get response
        chat_result = get_chat_response(transcription_text, language)

        return jsonify({
            'transcription': transcription_text,
            'response': chat_result['response'],
            'assessment': chat_result.get('assessment'),
            'language': language,
            'success': True
        })

    except Exception as e:
        logger.error(f"❌ Voice chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/text/chat', methods=['POST'])
def text_chat():
    """Text-based chat"""
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Detect language
        language = detect_language(user_message)

        # Get response
        chat_result = get_chat_response(user_message, language)

        return jsonify(chat_result)

    except Exception as e:
        logger.error(f"❌ Text chat error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/symptom/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms only (no chat response)"""
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Detect language if not provided
        if not language:
            language = detect_language(text)

        # Analyze symptoms
        assessment = symptom_checker.process_text(text, language)

        return jsonify({
            'symptoms': assessment.detected_symptoms,
            'severity': assessment.severity.value,
            'confidence': assessment.confidence,
            'recommendation_type': assessment.recommendation_type.value,
            'recommendations': assessment.recommendations,
            'explanation': assessment.raw_response,
            'language': assessment.language
        })

    except Exception as e:
        logger.error(f"❌ Symptom analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'status': 'ok',
        'whisper': 'available',
        'symptom_checker': 'available',
        'supported_languages': ['en', 'fr', 'ar'],
        'audio_formats': ['webm', 'mp3', 'wav', 'ogg']
    })


# -------------------- Run Server --------------------
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🎗️  BREAST CANCER SUPPORT CHATBOT")
    logger.info("=" * 60)
    logger.info("✅ Groq Whisper: Enabled")
    logger.info("✅ Symptom Checker: Ready")
    logger.info("✅ Languages: English, French, Arabic")
    logger.info("🚀 Starting server on http://127.0.0.1:5000")
    logger.info("=" * 60)

    app.run(host='127.0.0.1', port=5000, debug=False)