from ultralytics import YOLO
import cv2
import os
from gpiozero import Servo
from time import sleep

# Define recyclable and landfill categories
RECYCLABLE_ITEMS = [
    'bottle-glass',  # Glass bottles - recyclable
    'bottle-plastic',  # Plastic bottles - recyclable
    'tin can',  # Metal/aluminum - recyclable
    'gym bottle'  # Reusable water bottles - recyclable
]

LANDFILL_ITEMS = [
    'cup-disposable',  # Disposable cups - landfill
    'glass-wine',  # Broken wine glass - landfill (too fragile)
    'glass-normal',  # Broken drinking glass - landfill
    'glass-mug',  # Broken ceramic mug - landfill
    'cup-handle'  # Broken ceramic cup - landfill
]

# Human detection - update these to match your actual class names
HUMAN_CLASSES = [
    'person',  # Standard YOLO class name
    'human',
    'people',
    'man',
    'woman'
]


def classify_waste_type(class_name):
    """Classify detected object as recyclable, landfill, or person"""
    class_lower = class_name.lower()

    # Check for humans FIRST (priority)
    if class_lower in [h.lower() for h in HUMAN_CLASSES]:
        return "PERSON", (255, 0, 255)  # Purple/Magenta for people
    elif class_lower in [item.lower() for item in RECYCLABLE_ITEMS]:
        return "RECYCLABLE", (0, 255, 0)  # Green
    elif class_lower in [item.lower() for item in LANDFILL_ITEMS]:
        return "LANDFILL", (0, 0, 255)  # Red
    else:
        return "UNKNOWN", (255, 255, 0)  # Yellow


def train_model():
    """Train the YOLO model"""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    model = YOLO('yolo11n.pt')
    model.train(
        data='D:/Myself/University/EE/Trashcan/data.yaml',
        epochs=300,
        imgsz=640,
        device=0,
    )

    print("\nTraining completed!")
    print("Trained model saved to: runs/detect/train/weights/best.pt")


def run_webcam_detection(model_path='runs/detect/train33/weights/best.pt'):
    """Run real-time waste detection from webcam"""
    print("=" * 60)
    print("WEBCAM DETECTION MODE")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please update the model path or train a model first.")
        return

    # Load trained model
    model = YOLO(model_path)
    print(f"\nModel loaded successfully!")
    print(f"Detected classes: {model.names}")
    print("\nChecking for human detection capability...")

    # Check if model can detect humans
    has_human_class = any(
        any(human.lower() in str(class_name).lower() for human in HUMAN_CLASSES)
        for class_name in model.names.values()
    )

    if has_human_class:
        print("✓ Human detection enabled")
    else:
        print("⚠ Warning: No human class found in model")
        print("  Model may not detect people")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nStarting waste detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    print("-" * 60)


    try:
        RECYCLABLE_SERVO_PIN = 17 #define GPIO
        LANDFILL_SERVO_PIN = 18 #define GPIO
        
        servo_left = Servo(RECYCLABLE_SERVO_PIN)  #call object in Servo class
        servo_right = Servo(LANDFILL_SERVO_PIN)  #call object in Servo class

        #Set up 2 servos to it minimum position (0 degree)
        servo_left.min() 
        servo_right.min()
        sleep(1)
        servo_enabled = True
    except Exception as e:
        print(f"Error initializing GPIO: {e}")
        print("Servo control will be disabled")
        servo_enabled = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Reset counters for current frame
            recyclable_count = 0
            landfill_count = 0
            person_count = 0
            unknown_count = 0

            # Perform inference
            results = model(frame, conf=0.5, device=0, verbose=False)

            # Process detections
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class name and confidence
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0])

                    # Classify waste type
                    waste_type, color = classify_waste_type(class_name)

                    # Update counters
                    if waste_type == "RECYCLABLE":
                        recyclable_count += 1
                    elif waste_type == "LANDFILL":
                        landfill_count += 1
                    elif waste_type == "PERSON":
                        person_count += 1
                    else:
                        unknown_count += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Create label
                    label = f"{class_name} ({waste_type}) {confidence:.0%}"

                    # Draw label background
                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        frame, (x1, y1 - label_h - 8),
                        (x1 + label_w + 4, y1), color, -1
                    )

                    # Draw label text
                    cv2.putText(
                        frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )

            # Display statistics overlay
            stats_bg_height = 140
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (280, stats_bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Draw statistics
            cv2.putText(
                frame, f"Recyclable: {recyclable_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Landfill: {landfill_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                frame, f"People: {person_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )
            cv2.putText(
                frame, f"Unknown: {unknown_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            #logic servo
            if servo_enabled:
                if recyclable_count > 0 and landfill_count == 0: #detect recyclable
                    servo_left.max() #rotate left servo 90
                    servo_right.min() #keep the right servo at 0
                elif recyclable_count == 0 and landfill_count > 0: #detect landfill
                    servo_left.min() #keep the left servo at 0
                    servo_right.max() #rotate the right servo 90
                elif recyclable_count == 0 and landfill_count == 0: #detect human
                    servo_left.min() #keep the left servo at 0
                    servo_right.min() #keep the right servo at 0
            


            # Show frame
            cv2.imshow('Waste Classification with Human Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = 'waste_detection_screenshot.jpg'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        if servo_enabled:
            print("Detaching servo\nCleaning up GPIO")
            servo_left.min()
            servo_right.min()
            sleep(1)
            servo_left.detach()
            servo_right.detach()
            print("GPIO cleaned up")
        print("Webcam detection ended.")


if __name__ == '__main__':
    # Set to True to train, False to run detection
    TRAIN_MODE = False

    if TRAIN_MODE:
        train_model()
    else:
        # Update this path to your latest trained model
        run_webcam_detection('runs/detect/train33/weights/best.pt')