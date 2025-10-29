from ultralytics import YOLO
import cv2
import os
#from gpiozero import Servo, Button
import gpiozero
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
        GATE_SERVO_1 = 19
        GATE_SERVO_2 = 20
        GATE_SERVO_3 = 21
        GATE_SERVO_4 = 22
        START_BUTTON_PIN = 26
        
        #call object in Servo class, need to test pulse_width_max and min
        servo_recycle = Servo(RECYCLABLE_SERVO_PIN)  
        servo_landfill = Servo(LANDFILL_SERVO_PIN)  
        gate_servo_1 = Servo(GATE_SERVO_1)
        gate_servo_2 = Servo(GATE_SERVO_2)
        gate_servo_3 = Servo(GATE_SERVO_3)
        gate_servo_4 = Servo(GATE_SERVO_4)
        start_button = Button(START_BUTTON_PIN)

        

        #Set up 2 servos to it minimum position (0 degree)
        servo_recycle.min() 
        servo_landfill.min()
        gate_servo_1.mid()
        gate_servo_2.mid()
        gate_servo_3.mid()
        gate_servo_4.mid()
        
        sleep(1)
        servo_enabled = True
    except Exception as e:
        print(f"Error initializing GPIO: {e}")
        print("Servo control will be disabled")
        servo_enabled = False
    #state machine:
    # 0 = IDLE
    current_state = 0

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
            is_object_detected = (recyclable_count > 0) or (landfill_count > 0)
            if servo_enabled:
                #state 0: wait to press the button, trigger the first gate
                if current_state == 0:
                    if start_button.is_pressed:
                        current_state = 1  #turn to state 1
                #state 1: open the gate from 90 to 0     
                elif current_state == 1:
                    #drop 1 object for the first time
                    gate_servo_1.min()
                    gate_servo_2.min()
                    gate_servo_3.min()
                    gate_servo_4.min()

                    sleep(1)
                    #close the gate
                    gate_servo_1.mid()
                    gate_servo_2.mid()
                    gate_servo_3.mid()
                    gate_servo_4.mid()

                    current_state = 2  #turn to state 2
                #state 2: check if the camera detect the object or not
                elif current_state == 2: 
                    if is_object_detected:
                        current_state = 3  #turn to state 3
                #state 3: sort item
                elif current_state == 3:  
                    if recyclable_count > 0 and landfill_count == 0: #detect recyclable
                        servo_recycle.max() #rotate recycle servo 90
                        servo_landfill.min() #keep the landfill servo at 0
                        sleep(2)  #wait for object to fall
                        servo_recycle.min()  #close the recycle servo
                    elif recyclable_count == 0 and landfill_count > 0: #detect landfill
                        servo_recycle.min() #keep the recycle servo at 0
                        servo_landfill.max() #rotate the landfill servo 90
                        sleep(2)  #wait for object to fall
                        servo_landfill.min()  #close the landfill servo
                    elif recyclable_count == 0 and landfill_count == 0: #detect human
                        servo_recycle.min() #keep the recycle servo at 0
                        servo_landfill.min() #keep the landfill servo at 0
                    current_state = 4  #turn to state 4
                #state 4: go back to 1 when the camerac is clear
                elif current_state == 4:
                    if recyclable_count == 0 and landfill_count == 0:  #the object is sorted out, camera is clear
                        current_state = 1
            
            state_text = ["IDLE", "DROP", "WAIT_DETECT", "SORT", "WAIT_CLEAR"]
            if current_state < len(state_text):
                cv2.putText(frame, f"State: {state_text[current_state]}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                


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
        # Clean up objects and resources
        cap.release()
        cv2.destroyAllWindows()
        if servo_enabled:
            print("Detaching servo\nCleaning up GPIO")
            servo_recycle.min()
            servo_landfill.min()
            sleep(1)
            servo_recycle.detach()
            servo_landfill.detach()
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