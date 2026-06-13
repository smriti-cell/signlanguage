import cv2
import mediapipe as mp
import joblib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__, template_folder='.')

# =========================
# LANGUAGE SETTINGS & DATA
# =========================
global_language = "en"
global_prediction = ""
global_translation = "Show a hand..."
global_mode = "phrases"

GESTURES = [
    # Phrases (0-11)
    "Hello", "Yes", "No", "Thank You", "Sorry", "Good Morning", "OK", 
    "Help", "Please", "I Love You", "Stop", "Peace",
    # Alphabets (12-37)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

TRANSLATIONS = {
    "en": {
        "Hello": "Hello", "Yes": "Yes", "No": "No", "Thank You": "Thank You", "Sorry": "Sorry",
        "Good Morning": "Good Morning", "OK": "OK", "Help": "Help", "Please": "Please",
        "I Love You": "I Love You", "Stop": "Stop", "Peace": "Peace",
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H",
        "I": "I", "J": "J", "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P",
        "Q": "Q", "R": "R", "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X",
        "Y": "Y", "Z": "Z"
    },
    "hi": {
        "Hello": "नमस्ते", "Yes": "हाँ", "No": "नहीं", "Thank You": "धन्यवाद", "Sorry": "माफ़ कीजिए",
        "Good Morning": "सुप्रभात", "OK": "ठीक है", "Help": "मदद", "Please": "कृपया",
        "I Love You": "मुझे तुमसे प्यार है", "Stop": "रुकिए", "Peace": "शांति",
        "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच",
        "I": "आई", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी",
        "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू", "V": "वी", "W": "डब्लू", "X": "एक्स",
        "Y": "वाई", "Z": "जेड"
    },
    "kn": {
        "Hello": "ನಮಸ್ಕಾರ", "Yes": "ಹೌದು", "No": "ಇಲ್ಲ", "Thank You": "ಧನ್ಯವಾದಗಳು", "Sorry": "ಕ್ಷಮಿಸಿ",
        "Good Morning": "ಶುಭೋದಯ", "OK": "ಸರಿ", "Help": "ಸಹಾಯ", "Please": "ದಯವಿಟ್ಟು",
        "I Love You": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ", "Stop": "ನಿಲ್ಲಿಸಿ", "Peace": "ಶಾಂತಿ",
        "A": "ಎ", "B": "ಬಿ", "C": "ಸಿ", "D": "ಡಿ", "E": "ಇ", "F": "ಎಫ್", "G": "ಜಿ", "H": "ಎಚ್",
        "I": "ಐ", "J": "ಜೆ", "K": "ಕೆ", "L": "ಎಲ್", "M": "ಎಮ್", "N": "ಎನ್", "O": "ಒ", "P": "ಪಿ",
        "Q": "ಕ್ಯು", "R": "ಆರ್", "S": "ಎಸ್", "T": "ಟಿ", "U": "ಯು", "V": "ವಿ", "W": "ಡಬ್ಲ್ಯೂ", "X": "ಎಕ್ಸ್",
        "Y": "ವೈ", "Z": "ಝೆಡ್"
    },
    "ta": {
        "Hello": "வணக்கம்", "Yes": "ஆம்", "No": "இல்லை", "Thank You": "நன்றி", "Sorry": "மன்னிக்கவும்",
        "Good Morning": "காலை வணக்கம்", "OK": "சரி", "Help": "உதவி", "Please": "தயவுசெய்து",
        "I Love You": "நான் உன்னை காதலிக்கிறேன்", "Stop": "நில்லுங்கள்", "Peace": "அமைதி",
        "A": "ஏ", "B": "பி", "C": "சி", "D": "டி", "E": "இ", "F": "எஃப்", "G": "ஜி", "H": "எச்",
        "I": "ஐ", "J": "ஜே", "K": "கே", "L": "எல்", "M": "எம்", "N": "என்", "O": "ஓ", "P": "பி",
        "Q": "கியூ", "R": "ஆர்", "S": "எஸ்", "T": "டி", "U": "யூ", "V": "வி", "W": "டபிள்யூ", "X": "எக்ஸ்",
        "Y": "ஒய்", "Z": "இசட்"
    },
    "te": {
        "Hello": "నమస్కారం", "Yes": "అవును", "No": "కాదు", "Thank You": "ధన్యవాదాలు", "Sorry": "క్షమించండి",
        "Good Morning": "శుభోదయం", "OK": "సరే", "Help": "సహాయం", "Please": "దయచేసి",
        "I Love You": "నేను నిన్ను ప్రేమిస్తున్నాను", "Stop": "ఆపండి", "Peace": "శాంతి",
        "A": "ఏ", "B": "బీ", "C": "సీ", "D": "డీ", "E": "ఈ", "F": "ఎఫ్", "G": "జీ", "H": "హెచ్",
        "I": "ఐ", "J": "జే", "K": "కే", "L": "ఎల్", "M": "ఎమ్", "N": "ఎన్", "O": "ఓ", "P": "పీ",
        "Q": "క్యూ", "R": "ఆర్", "S": "ఎస్", "T": "టీ", "U": "యూ", "V": "వీ", "W": "డబ్ల్యూ", "X": "ఎక్స్",
        "Y": "వై", "Z": "జెడ్"
    },
    "mr": {
        "Hello": "नमस्कार", "Yes": "हो", "No": "नाही", "Thank You": "धन्यवाद", "Sorry": "माफ करा",
        "Good Morning": "शुभ सकाळ", "OK": "ठीक आहे", "Help": "मदत", "Please": "कृपया",
        "I Love You": "माझे तुझ्यावर प्रेम आहे", "Stop": "थांबा", "Peace": "शांतता",
        "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच",
        "I": "आय", "J": "जे", "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी",
        "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी", "U": "यू", "V": "व्ही", "W": "डब्लू", "X": "एक्स",
        "Y": "वाय", "Z": "झेड"
    },
    "bn": {
        "Hello": "নমস্কার", "Yes": "হ্যাঁ", "No": "না", "Thank You": "ধন্যবাদ", "Sorry": "দুঃখিত",
        "Good Morning": "সুপ্রভাত", "OK": "ঠিক আছে", "Help": "সাহায্য", "Please": "দয়া করে",
        "I Love You": "আমি তোমাকে ভালোবাসি", "Stop": "থামুন", "Peace": "শুদ্ধি",
        "A": "এ", "B": "বি", "C": "সি", "D": "ডি", "E": "ই", "F": "এফ", "G": "জি", "H": "এইচ",
        "I": "আই", "J": "জে", "K": "কে", "L": "এল", "M": "এম", "N": "এন", "O": "ও", "P": "পি",
        "Q": "কিউ", "R": "আর", "S": "এস", "T": "টি", "U": "ইউ", "V": "ভি", "W": "ডাব্লু", "X": "এক্স",
        "Y": "ওয়াই", "Z": "জেড"
    },
    "es": {
        "Hello": "Hola", "Yes": "Sí", "No": "No", "Thank You": "Gracias", "Sorry": "Lo siento",
        "Good Morning": "Buenos días", "OK": "De acuerdo", "Help": "Ayuda", "Please": "Por favor",
        "I Love You": "Te amo", "Stop": "Detener", "Peace": "Paz",
        "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H",
        "I": "I", "J": "J", "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P",
        "Q": "Q", "R": "R", "S": "S", "T": "T", "U": "U", "V": "V", "W": "W", "X": "X",
        "Y": "Y", "Z": "Z"
    },
    "pa": {
        "Hello": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "Yes": "ਹਾਂ", "No": "ਨਹੀਂ", "Thank You": "ਧੰਨਵਾਦ", "Sorry": "ਮਾਫ਼ ਕਰਨਾ",
        "Good Morning": "ਸ਼ੁਭ ਸਵੇਰ", "OK": "ਠੀਕ ਹੈ", "Help": "ਮਦਦ", "Please": "ਕਿਰਪา ਕਰਕੇ",
        "I Love You": "ਮੈਂ ਤੈਨੂੰ ਪਿਆਰ ਕਰਦਾ ਹਾਂ", "Stop": "ਰੁਕੋ", "Peace": "ਸ਼ਾਂਤੀ",
        "A": "ਏ", "B": "ਬੀ", "C": "ਸੀ", "D": "ਡੀ", "E": "ਈ", "F": "ਐੱਫ", "G": "ਜੀ", "H": "ਐੱਚ",
        "I": "ਆਈ", "J": "ਜੇ", "K": "ਕੇ", "L": "ਐੱਲ", "M": "ਐੱਮ", "N": "ਐੱਨ", "O": "ਓ", "P": "ਪੀ",
        "Q": "ਕਿਊ", "R": "ਆਰ", "S": "ਐੱਸ", "T": "ਟੀ", "U": "ਯੂ", "V": "ਵੀ", "W": "ਡਬਲਯੂ", "X": "ਐਕਸ",
        "Y": "ਵਾਈ", "Z": "ਜੈੱਡ"
    },
    "gu": {
        "Hello": "નમસ્તે", "Yes": "હા", "No": "ના", "Thank You": "આભાર", "Sorry": "દિલગીર છું",
        "Good Morning": "સુપ્રભાત", "OK": "બરાબર", "Help": "મદદ", "Please": "મહેરબાની કરીને",
        "I Love You": "હું તને પ્રેમ કરું છું", "Stop": "થોભો", "Peace": "શાંતિ",
        "A": "એ", "B": "બી", "C": "સી", "D": "ડી", "E": "ઈ", "F": "એફ", "G": "જી", "H": "એચ",
        "I": "આઈ", "J": "જે", "K": "કે", "L": "એલ", "M": "એમ", "N": "એન", "O": "ઓ", "P": "પી",
        "Q": "ક્યૂ", "R": "આર", "S": "એસ", "T": "ટી", "U": "યુ", "V": "વી", "W": "ડબ્લ્યુ", "X": "એક્સ",
        "Y": "વાય", "Z": "ઝેડ"
    }
}

# =========================
# FONT LOADER
# =========================
def get_font(lang, size=36):
    # Specific font files to try per language on Windows / locally
    fonts_by_lang = {
        "hi": ["C:\\Windows\\Fonts\\Nirmala.ttf", "NotoSansDevanagari-VariableFont_wdth,wght.ttf", "C:\\Windows\\Fonts\\mangal.ttf"],
        "mr": ["C:\\Windows\\Fonts\\Nirmala.ttf", "NotoSansDevanagari-VariableFont_wdth,wght.ttf", "C:\\Windows\\Fonts\\mangal.ttf"],
        "kn": ["C:\\Windows\\Fonts\\Nirmala.ttf", "NotoSansKannada-VariableFont_wdth,wght.ttf", "C:\\Windows\\Fonts\\tunga.ttf"],
        "ta": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\latha.ttf", "C:\\Windows\\Fonts\\vijaya.ttf"],
        "te": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\gautami.ttf", "C:\\Windows\\Fonts\\vani.ttf"],
        "bn": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\vrinda.ttf", "C:\\Windows\\Fonts\\shonar.ttf"],
        "pa": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\raavi.ttf"],
        "gu": ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\shruti.ttf"],
    }
    
    if lang in fonts_by_lang:
        for path in fonts_by_lang[lang]:
            try:
                return ImageFont.truetype(path, size)
            except IOError:
                continue
                
    # Fallback to Nirmala UI (common Indic font)
    for path in ["C:\\Windows\\Fonts\\Nirmala.ttf", "C:\\Windows\\Fonts\\nirmala.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except IOError:
            continue
            
    # Default fallback
    return ImageFont.load_default()

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]
    points = []

    for lm in hand.landmark:
        points.extend([lm.x, lm.y, lm.z])

    return points, hand

# =========================
# CAMERA ACCESS
# =========================
cap = None

def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def gen_frames():
    global global_prediction, global_translation, global_language, global_mode
    camera = get_camera()
    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            cv2.waitKey(100)
            continue

        frame_count += 1
        landmarks, hand_lms = extract_landmarks(frame)

        if hand_lms:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        if landmarks:
            # Only predict every 3rd frame to reduce lag
            if frame_count % 3 == 0:
                data = np.array(landmarks).reshape(1, -1)
                
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(data)[0]
                    class_probs = dict(zip(model.classes_, probs))
                    
                    if global_mode == "phrases":
                        allowed_labels = list(range(12))
                    else:
                        allowed_labels = list(range(12, 38))
                    
                    pred = max(allowed_labels, key=lambda l: class_probs.get(l, -1.0))
                else:
                    pred = model.predict(data)[0]
                
                # Safe boundary check
                if 0 <= pred < len(GESTURES):
                    gesture = GESTURES[pred]
                    display_text = TRANSLATIONS[global_language].get(gesture, gesture)
                else:
                    gesture = "Unknown"
                    display_text = "Unknown"

                global_prediction = gesture
                global_translation = display_text

            # Draw basic bounding box overlay on raw video stream
            cv2.rectangle(frame, (10, 10), (520, 90), (0, 0, 0), -1)

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)

            font = get_font(global_language, 36)
            draw.text((20, 25), display_text, font=font, fill=(0, 255, 0))

            frame = np.array(img_pil)

        else:
            global_prediction = ""
            global_translation = ""
            cv2.putText(
                frame,
                "Show a hand...",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                3
            )

        # Convert to JPEG bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return render_template('gesture.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_language', methods=['POST'])
def change_language():
    global global_language
    data = request.get_json()
    lang = data.get('lang', 'en')
    if lang in TRANSLATIONS:
        global_language = lang
        return jsonify({"status": "success", "language": global_language})
    return jsonify({"status": "error", "message": "Invalid language"}), 400

@app.route('/get_prediction')
def get_prediction():
    global global_prediction, global_translation, global_language
    return jsonify({
        "prediction": global_prediction,
        "translation": global_translation,
        "language": global_language
    })

@app.route('/change_mode', methods=['POST'])
def change_mode():
    global global_mode
    data = request.get_json()
    mode = data.get('mode', 'phrases')
    if mode in ['phrases', 'alphabets']:
        global_mode = mode
        return jsonify({"status": "success", "mode": global_mode})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

@app.route('/get_gestures')
def get_gestures():
    return jsonify({
        "gestures": GESTURES,
        "translations": TRANSLATIONS
    })

if __name__ == '__main__':
    print("Starting Sign Language Translator Flask App...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=False)
