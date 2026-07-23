# SignSpeak - Real-Time Sign Language Translator
SignSpeak is an AI-powered real-time Sign Language Translator built using Python, OpenCV, MediaPipe and Machine Learning.

It detects hand gestures through a webcam and instantly translates them into text and voice output in multiple Indian languages.

---

# Features
- Real-time webcam gesture detection
- Hand landmark tracking using MediaPipe
- Supports Sign Language Alphabets (A-Z)
- Supports Common Phrases
- Text-to-Speech voice output
- Multi-language translation
  - English
  - Hindi
  - Kannada
  - Tamil
  - Telugu
  - Marathi
  - Bengali
  - Punjabi
  - Gujarati
- Live session history
- Modern responsive web interface
---

# Tech Stack
- Python
- Flask
- OpenCV
- MediaPipe
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS
- JavaScript

---

# Project Structure
```
signlanguage/
│
├── app.py
├── collect_data.py
├── train_model.py
├── gesture_data.csv
├── model.pkl
├── requirements.txt
├── gesture.html
├── run_translator.bat
│
├── NotoSansDevanagari-VariableFont_wdth,wght.ttf
└── NotoSansKannada-VariableFont_wdth,wght.ttf
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/smriti-cell/signlanguage.git
```

Move into the project folder

```bash
cd signlanguage
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

---

# Usage
1. Launch the application.
2. Allow webcam access.
3. Show a sign language gesture.
4. The model predicts the gesture.
5. The detected gesture is translated into text.
6. Voice output is generated automatically.
7. Switch between supported languages anytime.
---

# Screenshots
## Home Interface

<img width="1600" height="899" alt="WhatsApp Image 2026-07-23 at 9 46 54 AM" src="https://github.com/user-attachments/assets/4f2f98a3-8b51-415c-b53a-f4b4fb62fa8c" />

---

## Alphabet Recognition

<img width="1600" height="899" alt="WhatsApp Image 2026-07-22 at 8 54 53 PM" src="https://github.com/user-attachments/assets/96371fa4-6e62-4f61-aa11-3f3f9e4cde7f" />

---

## Phrase Recognition

<img width="1600" height="899" alt="WhatsApp Image 2026-07-22 at 8 54 54 PM" src="https://github.com/user-attachments/assets/f191399f-db43-45e4-bcfe-55c37f88a27b" />

---

#  Future Improvements
- Sentence formation using multiple gestures
- Dynamic sign recognition
- Indian Sign Language dataset expansion
- Mobile application
- User authentication
- Cloud deployment

---

#  Author
**Smriti**
AI & Machine Learning Student

GitHub:
https://github.com/smriti-cell

---
