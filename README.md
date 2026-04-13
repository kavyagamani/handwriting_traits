# handwriting_traits
📌 Problem Statement
Understanding a person’s personality is important in psychology, education, and recruitment. Traditionally, personality is assessed through tests, interviews, or observations, which are:
Time-consuming
Subjective and biased
Not always accurate
Handwriting is a natural and unique way of expression that reflects a person’s behavior and mental state. Traditional handwriting analysis (graphology) depends on human interpretation, which can vary from person to person and lacks consistency.
👉 Therefore, there is a need for an automated, accurate, and unbiased system that can analyze handwriting using Machine Learning and Deep Learning.
This project solves that problem by building a system that:
Extracts handwriting features
Uses AI models to analyze patterns
Predicts personality traits objectively


📖 Overview
Handwriting AI is a web-based application that predicts personality traits from handwritten text using Artificial Intelligence and Deep Learning.
The system analyzes handwriting images and maps them to the Big Five Personality Traits:
Openness
Conscientiousness
Extraversion
Agreeableness
Neuroticism
✔️ Fast
✔️ Automated
✔️ Unbiased

🧠 Models Used
This project uses two models:
1️⃣ Handwriting Feature Model (handprediction.pkl)
Extracts handwriting characteristics
Identifies:
Letter size
Slant
Word spacing
Line spacing
Baseline
Works as a feature extraction layer
2️⃣ Personality Prediction Model (personality_prediction.h5)
Deep Learning model (CNN + MobileNetV2)
Takes processed image/features as input
Outputs:
Probability scores for all traits
Final predicted personality


✨ Features
📤 Upload handwritten image
🧹 Image preprocessing
🔍 Feature extraction using ML model
🤖 Personality prediction using CNN
📊 Graphical result display
⚡ Fast prediction (~1–2 seconds)
🔐 No data storage (privacy safe)


🏗️ System Workflow
🧠 Methodology
1. Data Collection
Handwritten images (scanned/mobile)
Categorized into Big Five traits
2. Preprocessing
Resize (224×224)
Normalize (0–1)
Noise removal
Data augmentation
3. Feature Extraction
Letter size
Slant angle
Word spacing
Line spacing
Baseline alignment
4. Model Training
CNN with MobileNetV2
80% training / 20% testing
Optimizer: Adam
Loss: Categorical Crossentropy
5. Prediction
Model outputs probabilities
Highest probability → Final trait


🛠️ Tech Stack
💻 Backend
Python
Flask
🤖 Machine Learning
TensorFlow
Keras
MobileNetV2
Scikit-learn
🖼️ Image Processing
OpenCV
NumPy
Pillow
🌐 Frontend
HTML
CSS
JavaScript


📂 Project Structure
Handwriting-AI/
│── dataset/
│── model/
│   ├── handprediction.pkl
│   ├── personality_prediction.h5
│── preprocessing/
│── feature_extraction/
│── static/
│── templates/
│── app.py
│── requirements.txt
│── README.md



⚙️ Installation & Running the Project
🔹 Step 1: Clone Repository
git clone https://github.com/your-username/handwriting-ai.git
cd handwriting-ai
🔹 Step 2: Create Virtual Environment
python -m venv venv
👉 Activate Environment
Windows:
venv\Scripts\activate
Linux / Mac:
source venv/bin/activate
🔹 Step 3: Install Dependencies
pip install -r requirements.txt
🔹 Step 4: Add Model Files
Ensure these files exist inside /model/:
handprediction.pkl
personality_prediction.h5
🔹 Step 5: Run the Application
python app.py
🔹 Step 6: Open in Browser
http://localhost:5000
'
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix


🧪 Testing
✅ Unit Testing
✅ Integration Testing
✅ Edge Case Testing
✅ Performance Testing
✅ Usability Testing

⏱️ Response Time: ~1–2 seconds

🔒 Advantages
No human bias
Fast and automated
Easy to use
Works with different handwriting styles
Privacy-friendly

⚠️ Limitations
Depends on image quality
Limited dataset
Similar traits may overlap

🚀 Future Enhancements
📱 Mobile app
🌍 Multi-language support
📈 Larger dataset
🧠 Advanced deep learning models

📸 Screenshots
<img width="600" height="300" alt="Screenshot (530)" src="https://github.com/user-attachments/assets/661c22a4-7c2e-4245-984d-6587bc74c3e7" />
<img width="600" height="300" alt="Screenshot (531)" src="https://github.com/user-attachments/assets/f7d311be-caf8-4434-ad81-aa18f9ddf6f5" />
<img width="600" height="300" alt="Screenshot (533)" src="https://github.com/user-attachments/assets/1ed6215d-1d23-485a-8530-2db642767b4e" />
<img width="600" height="300" alt="Screenshot (534)" src="https://github.com/user-attachments/assets/dfd520e6-d638-40c7-b9a3-3781f0fbafdd" />


👩‍💻 Author
Kavya Hegde
MCA – Acharya Institute of Technology
