# PlantDocBot: AI-Enabled Plant Disease Diagnosis

PlantDocBot is a multimodal, AI-driven assistant designed for early crop disease detection [1]. By leveraging computer vision and conversational interfaces, the system allows users to upload images of infected leaves or describe symptoms textually to receive immediate diagnostic results and actionable treatment recommendations [1].

The system mitigates the "domain gap" common in laboratory-trained models by utilizing datasets designed for real-world, in-situ field conditions (such as varied lighting, complex backgrounds, and organic leaf damage) [1].

## 🚀 Features

* **Dual-Input Diagnostics:** Supports both image uploads (leaf scans) and textual symptom descriptions [2].
* **Robust Field Performance:** Optimized to handle natural background noise, lighting shifts, and varied leaf orientations [2].
* **Actionable Insights:** Provides not just a disease classification, but automated, practical treatment and mitigation steps [2].
* **Interactive Dashboard:** Built with a clean, lightweight user interface for rapid prototyping and real-time inference [2].

## 🛠 Architecture & Tech Stack

* **Frontend & Deployment:** Streamlit is used to build the interactive web dashboard and decision support system [3]. The user interface also incorporates CSS (16.0%), HTML (12.4%), and JavaScript (8.9%) [4].
* **Core Languages & Libraries:** Powered primarily by Python (62.7% of the codebase), utilizing OpenCV for image preprocessing, along with NumPy and Pandas [3, 4].
* **Machine Learning & Vision:** Focused on deep computer vision classification models optimized to close the performance gap between ideal laboratory images and field-captured data [3].

## 📂 Project Structure

```text
PlantDocBot/
│
├── backend_server/     # Core application logic, API routing, and ML model inference [5]
├── frontend_server/    # Streamlit dashboard, UI components, and client-side logic [5]
├── .gitignore          # Ignored files and environment variables [5]
└── README.md           # Project documentation [5]
💻 Installation & Setup
1. Clone the repository:
git clone https://github.com/swanandisalive/PlantDocBot.git
cd PlantDocBot
2. Set up a virtual environment (Recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt 
4. Run the application:
streamlit run frontend_server/app.py 
