# Video-Based Face Recognition

This project detects whether a specific person appears in a video using a reference image.

## Features
- Extracts frames from video
- Detects faces using dlib (via face_recognition)
- Generates 128-D face embeddings
- Compares faces with a reference image
- Outputs match confidence

## Technologies Used
- Python
- OpenCV
- face_recognition (dlib)
- NumPy
- Pillow

## How It Works
1. A reference image is loaded and encoded
2. Video frames are extracted at intervals
3. Faces are detected in each frame
4. Embeddings are compared with the reference
5. System reports whether the person is present

## How to Run
```bash
pip install -r requirements.txt
python main.py
