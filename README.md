# Multivariate\_anomaly\_detection

Industrial Process Anomaly Detection & Recommendation System
Multivariate Time Series Anomaly Detection with Graph Autoencoder

A next-generation anomaly detection system for industrial or multivariate time-series processes. This system detects abnormal patterns across correlated variables and provides actionable insights through feature attribution and anomaly scoring.

---

## The Idea: Beyond Single-Feature Detection

Traditional anomaly detection evaluates each variable individually, often missing complex inter-variable relationships.
Our system uses a **Graph Autoencoder** to model correlations between features as edges, capturing which interactions contribute most to anomalies.

Instead of relying solely on thresholds, the model evaluates relationships across all variables, providing **accurate anomaly scoring (0–100%)** and highlighting the top contributing features.


---

## ✨ Key Features

* **Multivariate Anomaly Detection:** Detects abnormal patterns across multiple correlated process variables.
* **Graph Autoencoder Model:** Models relationships between features using edges and correlation strength.
* **Interactive Dashboard:** Real-time monitoring of anomaly score, top contributing features, and recommended actions. Includes a gauge visualization.
* **Feature Attribution:** Highlights the features responsible for anomalies using reconstruction error.

---

## 🛠️ Technical Approach

* **Language:** Python 3
* **Core Libraries:** `numpy`, `pandas`, `torch`, `torch_geometric`, `scikit-learn`, `plotly`, `streamlit`
* **Backend Model:** Graph Autoencoder (correlation-based edges, reconstruction error for anomaly scoring)
* **Preprocessing:** Timestamp validation, missing value handling, data normalization
* **Visualization:** Interactive dashboard with gauge/speedometer for anomaly score

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone <repository-link>
cd <repository-directory>
```

### 2. Set Up Python Environment

```bash
python -m venv venv
```

**Activate the environment:**

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Prepare Dataset

* Place your CSV files in the designated dataset folder.
* Ensure timestamps are valid and the data is cleaned.

---

### 5. Train the Model

```bash
python main.py
```

Trains the Graph Autoencoder on the defined “normal” period and saves the trained model and preprocessing objects.

---

### 6. Run Inference

```bash
python src/infer.py
```

Generates a CSV file with anomaly scores and top feature contributions.

---

### 7. Launch the Dashboard

```bash
streamlit run <dashboard-file>
```

Opens the dashboard for real-time anomaly detection with anomaly score, top contributing features, and recommended actions.
