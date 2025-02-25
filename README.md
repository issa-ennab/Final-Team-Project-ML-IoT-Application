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

- **Clean Dataset**:https://drive.google.com/drive/folders/1AseZPB1U5AOYPH8IhQkuUuzlKNsEbIma?usp=sharing

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
🔹 **Environment**: Jupyter Notebook, Google Colab, Local Machine (M2 Mac)

---

## 📌 Tasks & Assignments

| Task                          | Assigned To               |
| ----------------------------- | ------------------------- |
| Data Cleaning & Preprocessing | [Issa Ennab]              |
| LSTM Model Implementation     | [Ajmal Jalal]             |
| Random Forest Model           | [Mani Katuri]             |
| Third Model Implementation    | [Issa Ennab]              |
| Tableau Visualizations        | [Issa Ennab]              |
| IoT Design Diagram            | [?]                       |
| Model Comparison & Report     | [Mani Katuri]             |
| API Integration (FastAPI)     | [Ajmal Jalal]             |

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

## 🧾 Dataset Summary

The dataset used in this project consists of multiple sensor readings and labeled data collected from experiments conducted across different road surfaces and vehicle conditions. Each dataset represents a unique aspect of the vehicle’s motion, road type, and sensor environment. This summary provides an overview of the number of records, columns, and data quality aspects such as missing values and duplicates. This helps us better understand the scope of the data and guides our preprocessing and model development.

| Dataset Name   | Number of Rows | Number of Columns | Duplicates | Missing Values | Sample Columns                                                                     |
| -------------- | -------------- | ----------------- | ---------- | -------------- | ---------------------------------------------------------------------------------- |
| MPU Left       | 144036         | 28                | 0          | 0              | timestamp, acc_x_dashboard, acc_y_dashboard, ...                                   |
| MPU Right      | 144036         | 28                | 0          | 0              | timestamp, acc_x_dashboard, acc_y_dashboard, ...                                   |
| GPS            | 144036         | 20                | 0          | 1467           | timestamp, latitude, longitude, elevation, accuracy                                |
| Settings Left  | 3              | 24                | 0          | 1              | placement, address_mpu, address_ak, gyroscope_full_scale, accelerometer_full_scale |
| Settings Right | 3              | 24                | 0          | 1              | placement, address_mpu, address_ak, gyroscope_full_scale, accelerometer_full_scale |
| Labels         | 144036         | 14                | 144015     | 0              | paved_road, unpaved_road, dirt_road, cobblestone_road, asphalt_road                |
| GPS MPU Left   | 144036         | 32                | 0          | 0              | timestamp, acc_x_dashboard, acc_y_dashboard, ...                                   |
| GPS MPU Right  | 144036         | 32                | 0          | 0              | timestamp, acc_x_dashboard, acc_y_dashboard, ...                                   |

---

## 🚀 Future Improvements & Next Steps

- Train models with additional features.
- Experiment with **GRU models** for improved time-series forecasting.
- Deploy an interactive **prediction API**.
- Collect **real-time sensor data** for future projects.

---

📢 **Acknowledgment**: Thanks to [Jeferson Menegazzo] for making this dataset publicly available.
