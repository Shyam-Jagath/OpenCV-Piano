# 🎹 Virtual-Piano  

Virtual-Piano is an AI-powered virtual piano that lets you play piano in the air without needing a physical instrument.  
Using **computer vision**, it detects finger movements and simulates key presses, allowing users to create music seamlessly.  

---

## 🚀 Features  

- 🎵 **Air Gesture Control** – Play piano using only Finger movements.  
- 🔍 **Computer Vision-Based Detection** – Detects finger speed and motion to simulate key presses.  
- 🔢 **10 Simultaneous Notes** – Allows playing up to 10 notes at a time.  
- 👀 **Smart Octave Control** – Close your eyes for 3 seconds to shift octaves automatically.  
- 📹 **Real-time Tracking** – Uses webcam-based tracking for precise hand motion recognition.  

---

## 🛠️ Technologies Used  

- **Python** 🐍  
- **OpenCV** 🎥 (Computer Vision for motion detection)  
- **MediaPipe** ✋ (Hand tracking)  
- **PyGame** 🔊 (Audio processing)  
- **NumPy** 🔢 (Data processing)  

---

## 🎯 How It Works  

1. **Start the Application** – The webcam captures hand movements in real time.  
2. **Detect Finger Motion** – When fingers move downward at a certain speed, it is considered a key press.  
3. **Play Notes** – The system plays corresponding piano notes based on detected hand positions.  
4. **Switch Octaves** – Close your eyes for **3 seconds** to change octaves.  
5. **Enjoy Playing!** 🎶  

---

## 🖥️ Installation & Setup  
- Clone the repo to Your local machine.
- Make sure that your webcam connectd
- Finally to run the piano run `Launcher1.py` or `Launcher2.py` file.
- `Launcher1.py` for basic piano and `Launcher2.py` to shift Octaves

### 🔹 Prerequisites  
Ensure you have any version of **Python below 3.12** installed.  

### 🔹 Install Dependencies  
```bash
pip install opencv-python mediapipe numpy pygame
