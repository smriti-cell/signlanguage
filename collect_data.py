print("STARTING DATA COLLECTION SCRIPT...")
print("DEBUG: Importing modules...")

import cv2
print("DEBUG: cv2 imported OK")

import mediapipe as mp
print("DEBUG: mediapipe imported OK")

import pandas as pd
print("DEBUG: pandas imported OK")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Gesture mapping
GESTURES = {
    "1": ("Hello", 0),
    "2": ("Yes", 1),
    "3": ("No", 2),
    "4": ("Thank You", 3),
    "5": ("Sorry", 4),
    "6": ("Good Morning", 5),
    "7": ("OK", 6),
    "8": ("Help", 7),
    "9": ("Please", 8),
    "10": ("I Love You", 9),
    "11": ("Stop", 10),
    "12": ("Peace", 11),
}

# Add alphabets A-Z as keys "a"-"z" (labels 12-37)
for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    key = char.lower()
    GESTURES[key] = (char, 12 + i)

# Extract hand landmarks
def extract_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0]
    output = []
    for p in lm.landmark:
        output.extend([p.x, p.y, p.z])
    return output


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: CAMERA FAILED TO OPEN!")
    exit()

print("Camera ready.")
print("\n-------------------------------------------------------------")
print("                  SELECT GESTURE TO CAPTURE                  ")
print("-------------------------------------------------------------")
print("PHRASES:")
print("  1 = Hello         2 = Yes           3 = No            4 = Thank You")
print("  5 = Sorry         6 = Good Morning  7 = OK            8 = Help")
print("  9 = Please       10 = I Love You   11 = Stop         12 = Peace")
print("\nALPHABETS:")
print("  Type any letter a-z (e.g. 'a' for A, 'b' for B, etc.)")
print("-------------------------------------------------------------")
print("Press q to quit\n")

df = []

# Try to load existing data if file exists to append to it
try:
    existing_df = pd.read_csv("gesture_data.csv")
    df = existing_df.values.tolist()
    print(f"Loaded {len(df)} existing samples from gesture_data.csv")
except Exception:
    print("No existing gesture_data.csv found. Creating new dataset.")

while True:
    key = input("Enter gesture key (1-12 or a-z) or 'q' to finish: ").strip().lower()

    if key == "q":
        break

    if key not in GESTURES:
        print("Invalid key! Please try again.")
        continue

    gesture_name, label = GESTURES[key]

    print(f"\n--- Get ready for gesture: {gesture_name} (Label: {label}) ---")
    input("Press ENTER to start capturing 200 samples...")

    count = 0
    while count < 200:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw guidance info
        cv2.putText(frame, f"Capturing '{gesture_name}': {count}/200", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Keep hand in frame and move slowly", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            # Draw mediapipe hand landmarks on stream
            # Process rgb again for drawing overlay
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            if result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                
            df.append(landmarks + [label])
            count += 1

        cv2.imshow("Collecting Data (Press 'q' to abort gesture)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Aborted current gesture capture.")
            break

    print(f"Finished capturing: {gesture_name}!\n")

cap.release()
cv2.destroyAllWindows()

if len(df) > 0:
    df_pd = pd.DataFrame(df)
    df_pd.to_csv("gesture_data.csv", index=False)
    print(f"Successfully saved {len(df_pd)} total samples to gesture_data.csv!")
else:
    print("No data collected.")
