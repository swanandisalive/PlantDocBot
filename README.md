# PlantDocBot: AI-Enabled Plant Disease Diagnosis

PlantDocBot is a multimodal, AI-driven assistant designed for early crop disease detection. By leveraging computer vision and conversational interfaces, the system allows users to upload images of infected leaves or describe symptoms textually to receive immediate diagnostic results and actionable treatment recommendations. 

The system mitigates the "domain gap" common in laboratory-trained models by utilizing datasets designed for real-world, in-situ field conditions (such as varied lighting, complex backgrounds, and organic leaf damage).

---

## 🚀 Features

* **Dual-Input Diagnostics:** Supports both image uploads (leaf scans) and textual symptom descriptions.
* **Robust Field Performance:** Optimized to handle natural background noise, lighting shifts, and varied leaf orientations.
* **Actionable Insights:** Provides not just a disease classification, but automated, practical treatment and mitigation steps.
* **Interactive Dashboard:** Built with a clean, lightweight user interface for rapid prototyping and real-time inference.

---

## 🛠️ Architecture & Tech Stack

* **Frontend & Deployment:** [Streamlit](https://streamlit.io/) — used to build the interactive web dashboard and decision support system.
* **Core Languages & Libraries:** Python, OpenCV (image preprocessing), NumPy, and Pandas.
* **Machine Learning & Vision:** Focused on deep computer vision classification models optimized to close the performance gap between ideal laboratory images and field-captured data.

---

## 📂 Project Structure

```text
├── assets/               # UI components, images, and static graphics
├── src/                  # Main application source code
│   ├── app.py            # Streamlit dashboard entry point
│   ├── model_loader.py   # Code handling inference and preprocessing
│   └── utils.py          # Helper functions for processing data
├── models/               # Trained model binaries/weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

⚙️ Getting Started
Prerequisites
Ensure you have Python 3.9+ installed on your system.

Installation
1. Clone the repository:
git clone [https://github.com/swanandisalive/PlantDocBot.git](https://github.com/swanandisalive/PlantDocBot.git)
cd PlantDocBot

2. Create and activate a virtual environment:
# Windows
   python -m venv venv
   .\venv\Scripts\activate

# macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
  pip install -r requirements.txt

Running the Application
To launch the interactive diagnostic dashboard locally, run:
  streamlit run src/app.py

Open your browser and navigate to http://localhost:8501 to interact with the bot.

📊 Dataset & Methodology
Standard datasets often assume pristine conditions (uniform backgrounds, single centered leaves). PlantDocBot focuses on in-situ diagnostic robustness by accounting for real field conditions.

Image Preprocessing: Images are resized, normalized, and augmented to match the input specifications of the underlying deep convolutional network.

Classification & Inference: The input features are extracted via a specialized computer vision pipeline to output highly precise disease categories and matching confidence scores.

🤝 Contributing
Contributions to improve classification metrics, add support for more plant species, or optimize the user interface are welcome.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.
