# %%
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from dataclasses import dataclass
import os  
import winsound

current_dir = os.path.abspath('.')

# %%
# Constants and configuration
@dataclass
class IDCardSpecs:
    aspect_ratio: float
    template_features: dict
    
    @staticmethod
    def create_standard_id():
        return IDCardSpecs(
            aspect_ratio=1.58,  # Standard ID card aspect ratio
            template_features={
                "logo": (0.05, 0.05, 0.3, 0.2),     # x1, y1, x2, y2 in percentage
                "photo_area": (0.7, 0.2, 0.95, 0.7), # x1, y1, x2, y2 in percentage
            }
        )
def sound_play(sound_path):
    """Play a sound file using winsound"""
    try:
        winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception as e:
        print(f"Error playing sound: {e}")

# Face detection functions
def init_face_detector():
    """Initialize face detector with option for lightweight detection"""
    # Instead of hardcoding file paths, check for file existence or use a more reliable method
    try:
        model_file = "opencv_face_detector_uint8.pb"
        config_file = "opencv_face_detector.pbtxt"
        detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        return {'type': 'opencv_dnn', 'model': detector}
    except Exception as e:
        print(f"Error initializing face detector: {e}")
        return None


def detect_faces(detector, image, min_confidence=0.92, min_face_size=20):
    """Detect faces in an image with performance optimizations"""
    # Resize image for faster processing (optional)
    scale_factor = 1.0  # Reduce to 0.5 for even faster processing
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w*scale_factor), int(h*scale_factor)))
    
    # OpenCV DNN-based detection
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    detector['model'].setInput(blob)
    detections = detector['model'].forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Convert to MTCNN-compatible format
            faces.append({
                'box': [x1, y1, x2-x1, y2-y1],
                'confidence': float(confidence),
                'keypoints': {}  # No keypoints with this detector
            })
    return faces

def draw_face(image, faces):
    """Draw rectangle around detected face"""
    if not faces:
        return image
    
    display_image = image.copy()
    x, y, w, h = faces[0]['box']
    confidence = faces[0]['confidence']
    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = f"{w} x {h}, conf: {confidence:.2f}"
    cv2.putText(display_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return display_image

# Face comparison functions
def init_face_analysis(det_size=(640, 640)):
    """Initialize face analysis with only necessary models for embedding extraction"""
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def get_face_embedding(app, image):
    """Extract face embedding from image"""
    faces = app.get(image)
    if len(faces) == 0:
        return None
    face = faces[0]
    if hasattr(face, 'det_score') and face.det_score < 0.5:
        print(f"Warning: Low quality face detection (score: {face.det_score:.2f})")
    
    return face.embedding

def compare_face_embeddings(feat1, feat2):
    """Compare two face embeddings and return similarity score"""
    if feat1 is None or feat2 is None:
        return -1.0  # Return negative value to indicate invalid comparison
        
    # Normalize vectors to unit length
    feat1 = feat1 / np.linalg.norm(feat1)
    feat2 = feat2 / np.linalg.norm(feat2)
    
    # Calculate cosine similarity
    sim = np.dot(feat1, feat2)
    return sim

def is_same_person(feat1, feat2, threshold=0.3):
    """Check if two face embeddings belong to the same person"""
    if feat1 is None or feat2 is None:
        return False
    return compare_face_embeddings(feat1, feat2) > threshold

# ID card detection functions
def extract_reference_features(reference_image, card_specs):
    """Extract features from reference ID card image"""
    # Convert to grayscale for feature extraction
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Store reference aspect ratio and features
    h, w = reference_image.shape[:2]
    reference_features = {
        "keypoints": keypoints,
        "descriptors": descriptors,
        "aspect_ratio": float(w) / h,
        "width": w,
        "height": h
    }
    
    return reference_features

def detect_id_card(frame, reference, reference_features, card_specs, min_matches=15):
    """
    Detect ID card in frame using ORB feature matching with stability improvements
    """
    # Convert frame to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to improve feature detection
    gray = cv2.equalizeHist(gray)
    
    # Initialize ORB detector with more features and better parameters
    orb = cv2.ORB_create(
        nfeatures=2000,          # Increase number of features
        scaleFactor=1.2,         # Smaller scale factor for better multi-scale detection
        nlevels=8,               # More scale levels
        edgeThreshold=31,        # Avoid features at image borders
        firstLevel=0,
        WTA_K=2,
        patchSize=31,            # Larger patch size for more distinctive features
        fastThreshold=20         # Adjust FAST detector threshold
    )
    
    # Detect keypoints and compute descriptors for current frame
    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray, None)
    
    # If no features found, return early
    if descriptors_frame is None or len(keypoints_frame) < 8:
        return False, None, None
    
    # Create feature matcher - use KNN matcher instead of BFMatcher with crossCheck
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Match descriptors between reference and current frame using KNN
    # This gives us the 2 best matches for each descriptor
    matches = bf.knnMatch(reference_features["descriptors"], descriptors_frame, k=2)
    
    # Apply ratio test to filter good matches (Lowe's ratio test)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:  # Ratio test
                good_matches.append(m)
    
    # Check if we have enough good matches
    if len(good_matches) < min_matches:
        return False, None, None
    
    # Extract matched keypoints
    ref_pts = np.float32([reference_features["keypoints"][m.queryIdx].pt for m in good_matches])
    frame_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches])
    
    # Find homography to map reference points to frame points
    # Use more stringent RANSAC threshold
    H, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        return False, None, None
    
    # Count inliers (matches that fit the homography model)
    inlier_count = np.sum(mask)
    
    # Require a minimum number of inliers (stronger check)
    if inlier_count < min_matches * 0.7:
        return False, None, None
    
    # Get dimensions of reference image
    h, w = reference.shape[:2]
    
    # Define reference corners
    ref_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # Project reference corners to frame using homography
    frame_corners = cv2.perspectiveTransform(ref_corners, H)
    
    # Convert to integer points for drawing
    corners = np.int32(frame_corners)
    
    # Check if the quadrilateral is convex
    if not cv2.isContourConvex(corners):
        return False, None, None
    
    # Check if detected quadrilateral has reasonable aspect ratio
    detected_width = max(
        np.linalg.norm(corners[0][0] - corners[3][0]),
        np.linalg.norm(corners[1][0] - corners[2][0])
    )
    detected_height = max(
        np.linalg.norm(corners[0][0] - corners[1][0]),
        np.linalg.norm(corners[2][0] - corners[3][0])
    )
    
    if detected_height < 50 or detected_width < 50:  # Card is too small
        return False, None, None
        
    if detected_height == 0:
        return False, None, None
        
    detected_aspect = detected_width / detected_height
    
    # Check if aspect ratio is close to expected card aspect ratio
    aspect_tolerance = 0.2  # Tighter tolerance (was 0.3)
    if abs(detected_aspect - card_specs.aspect_ratio) > aspect_tolerance * card_specs.aspect_ratio:
        return False, None, None
    
    # Check for reasonable card area
    frame_height, frame_width = frame.shape[:2]
    total_area = frame_width * frame_height
    card_area = cv2.contourArea(corners)
    
    min_area_percentage = 0.02  # Card should occupy at least 2% of the frame
    max_area_percentage = 0.9   # Card shouldn't be more than 90% of the frame
    
    if card_area / total_area < min_area_percentage or card_area / total_area > max_area_percentage:
        return False, None, None
    
    # Create perspective transform to get warped view of the card
    # Target size for warped image
    target_w = 600
    target_h = int(target_w / card_specs.aspect_ratio)
    
    # Define destination points for perspective transform
    dst_pts = np.float32([
        [0, 0],
        [0, target_h - 1],
        [target_w - 1, target_h - 1],
        [target_w - 1, 0]
    ])
    
    # Find perspective transform matrix
    M = cv2.getPerspectiveTransform(frame_corners.reshape(4, 2).astype(np.float32), dst_pts)
    
    # Apply perspective transform to get warped image
    warped = cv2.warpPerspective(frame, M, (target_w, target_h))
    
    return True, corners, warped

def draw_card(frame, corners):
    """Draw detected card boundaries"""
    if corners is None:
        return frame
        
    display_frame = frame.copy()
    cv2.drawContours(display_frame, [corners], -1, (0, 255, 0), 3)
    return display_frame

# Main execution function
def main():
    # Add these variables at the beginning of main()
    stable_frames_required = 20     # Number of stable frames required
    stable_frame_counter = 0        # Counter for stable frames
    last_detection_state = False    # Previous frame detection state
    audio_played = False  # Flag to play sound only once per verification cycle
    sound_dir = os.path.join(current_dir, "sounds")

    # Initialize face detector and analyzer
    face_detector = init_face_detector()
    face_analyzer = init_face_analysis()
    
    # Initialize ID card detector with reference image - improved error handling
    # Replace the single reference image loading with:
    # Load reference for student ID card
    reference_path_student = os.path.join(current_dir, "reference_id_card.jpg")
    if not os.path.exists(reference_path_student):
        print(f"Student ID reference image not found at {reference_path_student}")
        return
    reference_student = cv2.imread(reference_path_student)
    if reference_student is None:
        print("Could not load student ID card reference image")
        return
    card_specs_student = IDCardSpecs.create_standard_id()
    reference_features_student = extract_reference_features(reference_student, card_specs_student)
    
    # Load reference for CCCD (citizen ID card)
    reference_path_cccd = os.path.join(current_dir, "reference_cccd.jpg")
    reference_cccd = None
    if os.path.exists(reference_path_cccd):
        reference_cccd = cv2.imread(reference_path_cccd)
        if reference_cccd is not None:
            # Adjust specs for citizen ID card
            card_specs_cccd = IDCardSpecs(
                aspect_ratio=1.58,  # Adjust if needed
                template_features={
                    "photo_area": (0.10, 0.15, 0.35, 0.70)  # Photo area on CCCD
                }
            )
            reference_features_cccd = extract_reference_features(reference_cccd, card_specs_cccd)
        else:
            print("Could not load CCCD reference image")
    
    # Create list of reference cards to detect
    references = [
        {"type": "Student ID", "image": reference_student, "features": reference_features_student, "specs": card_specs_student},
    ]
    
    # Add CCCD to references if available
    if reference_cccd is not None:
        references.append({
            "type": "CCCD", 
            "image": reference_cccd, 
            "features": reference_features_cccd, 
            "specs": card_specs_cccd
        })

    # Initialize camera - add error handling
    cap = cv2.VideoCapture(0)
    
    # State variables
    state = "DETECT_CARD"
    card_face_embedding = None
    live_face_embedding = None
    
    print("=== Face Verification System ===")
    print("Step 1: Detecting ID card automatically")
    sound_path = os.path.join(sound_dir, "step1.wav")
    sound_play(sound_path)      
    # Main loop
    running = True
    while running:
        # Safely capture frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create display frame to show instructions and results
        display = frame.copy()
        
        # Process key presses first - ensures we always catch the quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break
        elif key == ord('r') and state == "SHOW_RESULT":
            state = "DETECT_CARD"
            card_face_embedding = None
            live_face_embedding = None
            stable_frame_counter = 0
            audio_played = False  # Reset audio flag
            print("Restarting verification process")
            print("Step 1: Detecting ID card automatically")
            sound_path = os.path.join(sound_dir, "step1.wav")
            sound_play(sound_path)        

        # State machine for the verification process
        if state == "DETECT_CARD":  
            card_detected = False
            corners = None
            warped = None
            
            # Try to detect each card type
            for ref in references:
                card_detected, corners, warped = detect_id_card(frame, ref["image"], ref["features"], ref["specs"])
                if card_detected:
                    detected_specs = ref["specs"]
                    detected_type = ref["type"]
                    break
            
            if card_detected:
                # Draw card outline
                display = draw_card(display, corners)
                cv2.putText(display, f"{detected_type} Detected", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update frame counter if detection is consistent
                stable_frame_counter = stable_frame_counter + 1 if last_detection_state else 0
                
                # Auto-capture after specified number of stable frames
                if stable_frame_counter >= stable_frames_required:
                    # Extract face from ID card
                    coords = detected_specs.template_features["photo_area"]
                    x1, y1, x2, y2 = [int(c * d) for c, d in zip(coords, [warped.shape[1], 
                                                                warped.shape[0], 
                                                                warped.shape[1], 
                                                                warped.shape[0]])]
                    card_face_img = warped[y1:y2, x1:x2]
                    
                    # Get face embedding from ID card
                    card_face_embedding = get_face_embedding(face_analyzer, card_face_img)
                    
                    if card_face_embedding is not None:
                        state = "DETECT_FACE"
                        stable_frame_counter = 0
                        print("Step 2: Detecting live face automatically")
                        sound_path = os.path.join(sound_dir, "step2.wav")
                        sound_play(sound_path)
                    else:
                        # Reset counter if face extraction failed
                        stable_frame_counter = 0
                        cv2.putText(display, "No face found on card, try again", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Move ID card into view", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                stable_frame_counter = 0
            
            # Update last detection state
            last_detection_state = card_detected
            
        elif state == "DETECT_FACE":

            faces = detect_faces(face_detector, frame)
            
            if faces:
                # Draw face rectangle
                display = draw_face(display, faces)
                cv2.putText(display, "Face Detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update frame counter if detection is consistent
                stable_frame_counter = stable_frame_counter + 1 if last_detection_state else 0
                
                # Auto-capture after specified number of stable frames
                if stable_frame_counter >= stable_frames_required:
                    # Get face embedding from live face
                    live_face_embedding = get_face_embedding(face_analyzer, frame)
                    
                    if live_face_embedding is not None:
                        state = "SHOW_RESULT"
                        stable_frame_counter = 0
                        print("Step 3: Comparing faces...")
                    else:
                        # Reset counter if face extraction failed
                        stable_frame_counter = 0
                        cv2.putText(display, "Face analysis failed, try again", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "Position your face in the camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                stable_frame_counter = 0
                
            # Update last detection state
            last_detection_state = len(faces) > 0
                
        elif state == "SHOW_RESULT":
            # Compare the face embeddings
            if card_face_embedding is not None and live_face_embedding is not None:
                similarity = compare_face_embeddings(card_face_embedding, live_face_embedding)
                is_match = is_same_person(card_face_embedding, live_face_embedding)
                
                # Display results - simplified
                result_color = (0, 255, 0) if is_match else (0, 0, 255)
                result_text = "MATCH VERIFIED" if is_match else "NO MATCH"
                
                cv2.putText(display, f"Result: {result_text}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                cv2.putText(display, f"Similarity: {similarity:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, "Press 'r' to restart, 'q' to quit", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Play sound if not already played
                if not audio_played:
                    sound_file = "match.wav" if is_match else "not_match.wav"
                    sound_path = os.path.join(sound_dir, sound_file)
                    sound_play(sound_path)
                    audio_played = True
            else:
                cv2.putText(display, "Error: Failed to extract face features", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, "Press 'r' to restart, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Face Verification System', display)
    
    # Clean up properly
    print("Exiting face verification system")
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

# %%



