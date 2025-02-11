# 🚗 Road Condition Prediction using Machine Learning

## 📌 Project Overview

This project explores **road conditions and vehicle behavior prediction** using machine learning techniques. The dataset contains sensor readings from multiple vehicles on different road types (asphalt, cobblestone, dirt) and includes GPS, accelerometer (MPU), and environmental data. Our goal is to build predictive models that classify road conditions and analyze vehicle behavior using **LSTM, Random Forest, and other models**.

📍 **Institution**: University of San Diego – IoT Course  
📍 **Instructor**: Anna Marbut  
📍 **Contributors**: Issa Ennab, Manikanta Katuri, Ajmal Jalal

---

## 📊 Dataset & Exploration

- **Source**: [Kaggle Dataset](https://www.kaggle.com/code/jefmenegazzo/pvs-data-exploration)
- **Collected from**: Three different vehicles, three road types, multiple sensors
- **Key Features**:
  - **GPS & Speed Data**: Tracks vehicle movement
  - **MPU Sensors (Left & Right)**: Measures vibration and motion
  - **Environmental Conditions**: Maps road surface conditions
  - **Road Type & Labels**: Indicates whether the road is cobblestone, asphalt, or dirt

---

## 🎯 Project Goals & Objectives

- **Predict road conditions** using sensor data.
- **Analyze vehicle speed behavior** across different terrains.
- **Classify vibration intensity** using MPU and GPS data.
- **Compare multiple ML models**:
  - 🔹 **LSTM (Long Short-Term Memory)** – Best for time-series and trajectory analysis.
  - 🔹 **Random Forest** – Effective for classification.
  - 🔹 **Third Model TBD** – Potentially another deep learning or traditional ML approach.
- **Use Tableau for Data Visualization** – Explore trends in sensor data.
- **Deploy as a REST API** – Utilize **FastAPI** to allow real-time predictions.

---

## 🛠️ Technical Stack

🔹 **Languages & Tools**: Python, TensorFlow/Keras, Scikit-learn, Pandas, NumPy  
🔹 **Libraries**: Matplotlib, Seaborn, FastAPI (for model serving)  
🔹 **Visualization**: Tableau, Plotly  
🔹 **Environment**: Jupyter Notebook, Google Colab, Local Machine (M1 Mac)

---

## 📌 Tasks & Assignments

| Task                          | Assigned To     |
| ----------------------------- | --------------- |
| Data Cleaning & Preprocessing | [Issa Ennab]    |
| LSTM Model Implementation     | [?]             |
| Random Forest Model           | [?]             |
| Third Model Implementation    | [?]             |
| Tableau Visualizations        | [Issa Ennab]    |
| Model Comparison & Report     | [?]             |
| API Integration (FastAPI)     | [?]             |
| PowerPoint & Presentation     | [Issa Ennab, ?] |

---

## 📥 Setup & Installation

1️⃣ **Clone the Repository**:

```bash
git clone https://github.com/issa-ennab/Final-Team-Project-ML-IoT-Application.git
cd Final-Team-Project-ML-IoT-Application
```

2️⃣ **Create Virtual Environment & Install Dependencies**:

```bash
python -m venv env
source env/bin/activate  # On Windows, use 'env\\Scripts\\activate'
pip install -r requirements.txt
```

3️⃣ **Run Jupyter Notebook**:

```bash
jupyter notebook
```

4️⃣ **Train & Evaluate Models**:

```bash
python train_lstm.py  # Example script to train LSTM model
python train_rf.py  # Train Random Forest model
```

---

## 📊 How to Use the Project

🔹 Run the models and compare outputs in Jupyter.  
🔹 Use Tableau to explore the dataset and predictions.  
🔹 Optionally, deploy models via FastAPI for external use.

---

## 🚀 Future Improvements & Next Steps

- Train models with additional features.
- Experiment with **GRU models** for improved time-series forecasting.
- Deploy an interactive **prediction API**.
- Collect **real-time sensor data** for future projects.

---

📢 **Acknowledgment**: Thanks to [Jeferson Menegazzo] for making this dataset publicly available.
