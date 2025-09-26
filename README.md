# VisionAI - An AI-Powered Visual Assistant

VisionAI is a Flutter-based mobile application designed to serve as a visual assistant for individuals with visual impairments. Inspired by apps like *Be My Eyes*, VisionAI aims to provide a fully independent solution, leveraging the power of on-device and cloud-based AI to describe the user's surroundings through a simple, hands-free, voice-activated interface.

## About The Project

The primary goal of VisionAI is to empower visually impaired users by providing them with real-time information about objects and text in their environment. The app uses a hybrid AI model to deliver a balance of speed and accuracy.  

- An initial, quick analysis is performed on the device for immediate feedback.  
- A more comprehensive description is delivered using a powerful cloud-based AI for richer context and higher accuracy.  

This project was built from the ground up, navigating complex challenges in mobile ML, including model conversion, on-device inference, and API integration.

## Features

- **Live Camera Feed**: A continuous and smooth camera preview.  
- **On-Demand Analysis**: Processing is triggered by a simple button press or voice command.  
- **Hybrid AI Model**:  
  - *On-Device*: Fast, offline object detection using a YOLOv8 model and text recognition with Google ML Kit.  
  - *Cloud-Powered*: Highly accurate, comprehensive scene descriptions using the Google Gemini API.  
- **Voice-Activated**: Hands-free control with voice commands like "capture" or "what's ahead".  
- **Text-to-Speech Output**: All results are spoken aloud in a clear, natural-sounding voice.  
- **Modern UI**: A clean, accessible user interface with glassmorphic elements.  

## Tech Stack

- **Framework**: Flutter  
- **Object Detection**: YOLOv8 (TFLite)  
- **Text Recognition**: Google ML Kit Text Recognition  
- **Cloud AI**: Google Gemini API (`google_generative_ai`)  
- **State Management**: setState  
- **Other Key Packages**: `camera`, `tflite_flutter`, `flutter_tts`, `speech_to_text`  

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- **Flutter SDK**: Ensure you have Flutter installed. You can follow the official guide [here](https://docs.flutter.dev/get-started/install).  
- **Code Editor**: VS Code with the Flutter extension or Android Studio.  
- **Google AI Studio API Key**: You will need an API key for the Gemini model. You can get one for free at [Google AI Studio](https://aistudio.google.com/).  

### Installation

1. **Clone the repo**  

git clone https://github.com/your_username/VisionAI.git
cd VisionAI

text

2. **Add ML Models and Assets**  
- Place your YOLOv8 model file in the following directory:  
  `assets/models/yolov8m.tflite`  
- Place the Tesseract data file (if using it as a fallback) in:  
  `assets/tessdata/eng.traineddata`  

3. **Set Up Your API Key**  
Create a `.env` file in the root directory of the project and add:  

GEMINI_API_KEY=YOUR_API_KEY_HERE

text

*Important*: The `.env` file is listed in `.gitignore` and should never be committed to the repository.  

4. **Install Packages**  

flutter pub get

text

5. **Run the App**  

flutter run

text

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. This is especially true for group members working on this project!

### How to Contribute

1. **Create a Branch**:  
git checkout -b feature/Add-New-Voice-Command

text

2. **Make Your Changes**: Implement your feature or bug fix.  

3. **Commit Your Changes**:  
git add .
git commit -m "feat: Add new voice command for weather"

text

4. **Push to Your Branch**:  
git push origin feature/Add-New-Voice-Command

text

5. **Open a Pull Request**:  
- Go to the GitHub repository.  
- Open a pull request from your new branch to the `main` branch.  
- Provide a clear description of the changes made.  

Your pull request will be reviewed, and once approved, it will be merged into the main project.

## Project Status

The project is currently a functional prototype. The hybrid AI model is implemented, and the core features are working.  

Future work will focus on:  
- Improving UI/UX  
- Refining model accuracy  
- Adding new features like a history of scans  
