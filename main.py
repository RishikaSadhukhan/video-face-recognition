import cv2
import face_recognition
import numpy as np
from PIL import Image
import os
from pathlib import Path


class VideoFaceRecognizer:
    def __init__(self, tolerance=0.6, frame_skip=5):
        """
        Initialize the face recognizer
        
        Args:
            tolerance: Lower values make matching more strict (default: 0.6)
            frame_skip: Process every Nth frame to speed up (default: 5)
        """
        self.tolerance = tolerance
        self.frame_skip = frame_skip
        self.reference_encoding = None
        self.reference_name = "reference_person"

    def load_reference_image(self, image_path):
        """
        Load and encode the reference image

        Args:
            image_path: Path to reference image file

        Returns:
            True if successful, False otherwise
        """
        print(f"Loading reference image: {image_path}")

        try:
            # Load image using PIL first to ensure clean loading
            from PIL import Image
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if pil_image.mode != 'RGB':
                print(f"Converting from {pil_image.mode} to RGB")
                pil_image = pil_image.convert('RGB')
            
            # Resize image if it's too large (dlib sometimes has issues with large images)
            max_dimension = 1024
            if max(pil_image.size) > max_dimension:
                print(f"Resizing large image from {pil_image.size}")
                ratio = max_dimension / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
                print(f"New size: {pil_image.size}")
            
            # Convert PIL image to numpy array with explicit dtype
            image = np.asarray(pil_image, dtype=np.uint8)
            
            # Make a copy to ensure it's writable and contiguous
            image = np.copy(image)
            
            # Double-check format
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"ERROR: Unexpected image shape: {image.shape}")
                return False
            
            print(f"Loaded shape: {image.shape} dtype: {image.dtype} contiguous: {image.flags['C_CONTIGUOUS']}")

            # Try with CNN model first (more robust), fallback to HOG
            try:
                print("Attempting face detection with CNN model...")
                face_locations = face_recognition.face_locations(image, model="cnn")
            except:
                print("CNN failed, trying HOG model...")
                face_locations = face_recognition.face_locations(image, model="hog")

            if len(face_locations) == 0:
                print("ERROR: No faces found in reference image")
                print("Try using a clearer image with a visible face")
                return False

            if len(face_locations) > 1:
                print(f"WARNING: Multiple faces found ({len(face_locations)}), using first one")

            # Get face encoding (128-dimensional vector)
            encodings = face_recognition.face_encodings(image, face_locations)
            self.reference_encoding = encodings[0]

            print(f"✓ Reference face encoded successfully")
            print(f"Encoding shape: {self.reference_encoding.shape}")
            print(f"Sample values: {self.reference_encoding[:5]}")

            return True

        except Exception as e:
            print(f"ERROR loading reference image: {e}")
            import traceback
            traceback.print_exc()
            
            # Try one more time with opencv as absolute fallback
            print("\nTrying alternative loading method with OpenCV...")
            try:
                img = cv2.imread(image_path)
                if img is None:
                    return False
                
                # Resize if needed
                height, width = img.shape[:2]
                if max(height, width) > 1024:
                    scale = 1024 / max(height, width)
                    img = cv2.resize(img, None, fx=scale, fy=scale)
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                if len(face_locations) == 0:
                    print("ERROR: No faces found")
                    return False
                
                encodings = face_recognition.face_encodings(rgb_img, face_locations)
                self.reference_encoding = encodings[0]
                print("✓ Successfully loaded using OpenCV fallback")
                return True
                
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return False

    def extract_frames(self, video_path, output_folder=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_folder: Optional folder to save frames
        
        Returns:
            List of frame arrays
        """
        print(f"\nExtracting frames from: {video_path}")
        
        # Open video
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print("ERROR: Could not open video")
            return []
        
        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        frames = []
        frame_count = 0
        saved_count = 0
        
        # Create output folder if specified
        if output_folder:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = video.read()
            
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % self.frame_skip == 0:
                frames.append(frame)
                
                # Optionally save frame
                if output_folder:
                    frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                saved_count += 1
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        video.release()
        print(f"Extracted {len(frames)} frames (every {self.frame_skip} frames)")
        
        return frames
    
    def detect_and_crop_faces(self, frames, output_folder=None):
        """
        Detect faces in frames and crop them
        
        Args:
            frames: List of frame arrays
            output_folder: Optional folder to save cropped faces
        
        Returns:
            List of tuples (frame_idx, face_location, face_image)
        """
        print("\nDetecting faces in frames...")
        
        detected_faces = []
        
        if output_folder:
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        for idx, frame in enumerate(frames):
            # Convert BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            
            for face_idx, face_location in enumerate(face_locations):
                top, right, bottom, left = face_location
                
                # Crop face from frame
                face_image = rgb_frame[top:bottom, left:right]
                
                detected_faces.append((idx, face_location, face_image))
                
                # Optionally save cropped face
                if output_folder:
                    face_path = os.path.join(output_folder, f"face_{idx}_{face_idx}.jpg")
                    face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_path, face_bgr)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(frames)} frames...")
        
        print(f"Detected {len(detected_faces)} faces across {len(frames)} frames")
        
        return detected_faces
    
    def generate_embeddings(self, frames, face_locations):
        """
        Generate face embeddings for detected faces
        
        Args:
            frames: List of frame arrays
            face_locations: List of (frame_idx, face_location, face_image)
        
        Returns:
            List of tuples (frame_idx, embedding, face_image)
        """
        print("\nGenerating face embeddings...")
        
        embeddings = []
        
        for frame_idx, face_loc, face_img in face_locations:
            # Get the frame
            frame = frames[frame_idx]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get encoding for this face
            face_encodings = face_recognition.face_encodings(rgb_frame, [face_loc])
            
            if len(face_encodings) > 0:
                embeddings.append((frame_idx, face_encodings[0], face_img))
        
        print(f"Generated {len(embeddings)} face embeddings")
        
        return embeddings
    
    def compare_faces(self, embeddings):
        """
        Compare detected faces with reference face
        
        Args:
            embeddings: List of (frame_idx, embedding, face_image)
        
        Returns:
            Dictionary with comparison results
        """
        if self.reference_encoding is None:
            print("ERROR: No reference encoding loaded")
            return None
        
        print(f"\nComparing faces with reference (tolerance: {self.tolerance})...")
        
        matches = []
        distances = []
        
        for frame_idx, encoding, face_img in embeddings:
            # Calculate face distance (lower = more similar)
            distance = face_recognition.face_distance([self.reference_encoding], encoding)[0]
            distances.append(distance)
            
            # Check if match
            is_match = distance <= self.tolerance
            
            if is_match:
                matches.append({
                    'frame_idx': frame_idx,
                    'distance': distance,
                    'confidence': 1 - distance,
                    'face_image': face_img
                })
        
        # Calculate statistics
        results = {
            'total_faces': len(embeddings),
            'matches': len(matches),
            'match_percentage': (len(matches) / len(embeddings) * 100) if embeddings else 0,
            'avg_distance': np.mean(distances) if distances else 0,
            'min_distance': np.min(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'matched_faces': matches,
            'is_person_present': len(matches) > 0
        }
        
        return results
    
    def process_video(self, video_path, reference_image_path, save_outputs=False):
        """
        Complete pipeline: process video and compare with reference
        
        Args:
            video_path: Path to video file
            reference_image_path: Path to reference image
            save_outputs: Save intermediate outputs to folders
        
        Returns:
            Dictionary with results
        """
        print("="*60)
        print("STARTING VIDEO FACE RECOGNITION PIPELINE")
        print("="*60)
        
        # Step 1: Load reference image
        if not self.load_reference_image(reference_image_path):
            return None
        
        # Step 2: Extract frames
        output_frames = "output/frames" if save_outputs else None
        frames = self.extract_frames(video_path, output_frames)
        
        if not frames:
            print("ERROR: No frames extracted")
            return None
        
        # Step 3: Detect and crop faces
        output_faces = "output/faces" if save_outputs else None
        face_detections = self.detect_and_crop_faces(frames, output_faces)
        
        if not face_detections:
            print("ERROR: No faces detected in video")
            return None
        
        # Step 4: Generate embeddings
        embeddings = self.generate_embeddings(frames, face_detections)
        
        if not embeddings:
            print("ERROR: Could not generate embeddings")
            return None
        
        # Step 5: Compare with reference
        results = self.compare_faces(embeddings)
        
        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Total frames processed: {len(frames)}")
        print(f"Total faces detected: {results['total_faces']}")
        print(f"Matching faces: {results['matches']}")
        print(f"Match percentage: {results['match_percentage']:.2f}%")
        print(f"Average face distance: {results['avg_distance']:.4f}")
        print(f"Best match distance: {results['min_distance']:.4f}")
        print(f"\nFINAL VERDICT: {'✓ PERSON FOUND IN VIDEO' if results['is_person_present'] else '✗ PERSON NOT FOUND IN VIDEO'}")
        print("="*60)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = VideoFaceRecognizer(
        tolerance=0.6,  # Adjust for stricter (lower) or looser (higher) matching
        frame_skip=5    # Process every 5th frame
    )
    
    # Paths (update these to your actual file paths)
    video_path = r"C:\Users\KIIT\Desktop\FaceRecognition\facerecognitionDeepface\QPXP1246.MP4"
    reference_image = r"C:\Users\KIIT\Desktop\FaceRecognition\facerecognitionDeepface\test_fixed.jpg"

    # Sanity check
    print("File exists?", os.path.exists(reference_image))
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        print("Please update the video_path variable with your actual video file")
    elif not os.path.exists(reference_image):
        print(f"ERROR: Reference image not found: {reference_image}")
        print("Please update the reference_image variable with your actual image file")
    else:
        # Process video
        results = recognizer.process_video(
            video_path=video_path,
            reference_image_path=reference_image,
            save_outputs=True  # Set to False to skip saving intermediate files
        )
        
        if results and results['matches'] > 0:
            print(f"\nFound {results['matches']} matching frames!")
            print("Best matches:")
            for i, match in enumerate(results['matched_faces'][:5], 1):
                print(f"  {i}. Frame {match['frame_idx']}, Confidence: {match['confidence']:.2%}")