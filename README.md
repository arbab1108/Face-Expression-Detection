# 🎭 Real-time Face Emotion Detection & Analytics

This project leverages deep learning and computer vision to perform **real-time facial emotion detection** using your webcam, visualizes emotion statistics dynamically with **Matplotlib**, and evaluates performance through a **confusion matrix** heatmap using synthetic data.

---

## 🔍 Features

- 🔵 **Live Emotion Detection** with bounding boxes on faces  
- 📊 **Real-time Emotion Statistics Dashboard** with auto-refresh using Matplotlib  
- 📈 **Confusion Matrix Visualization** for evaluating model predictions  
- 🤖 Powered by **MTCNN** for face detection and **DeepFace** for emotion recognition

---

## 🧠 Tech Stack

- Python 3.x  
- OpenCV  
- DeepFace  
- MTCNN  
- Matplotlib  
- Seaborn  
- scikit-learn  
- NumPy

---

## 📂 File Structure

```
├── emotion.py          # Real-time emotion detection and statistics visualization
├── matplotlib.py       # Confusion matrix heatmap generation using synthetic data
├── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/emotion-detection-analytics.git
cd emotion-detection-analytics
```

### 2. Install Dependencies

Ensure you have Python 3 installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install opencv-python deepface mtcnn matplotlib seaborn scikit-learn numpy
```

---

## ▶️ Running the Project

### Real-time Emotion Detection

```bash
python emotion.py
```

- Opens webcam, detects faces and displays detected emotions in real time.
- A separate window shows a dynamically updating emotion frequency chart.

### Confusion Matrix Visualization

```bash
python matplotlib.py
```

- Simulates predictions vs. ground truth using synthetic data.
- Displays a heatmap confusion matrix to evaluate classification performance.

---

## ✅ Future Improvements

- Integrate model predictions into confusion matrix dynamically  
- Add logging and analytics dashboard for long-term emotion tracking  
- Deploy on the web or mobile device  

---

## 🙌 Acknowledgements

- [DeepFace](https://github.com/serengil/deepface)  
- [MTCNN](https://github.com/ipazc/mtcnn)  
- [OpenCV](https://opencv.org/)  
