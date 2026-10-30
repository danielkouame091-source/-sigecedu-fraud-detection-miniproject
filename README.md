# SigecEdu Fraud/Blocking Detection Using Deep Learning (Neural Network)

## 1. Project Overview

Schools and school-management platforms often need to decide whether a student's account
should be temporarily **blocked** (restricted from services such as exam registration or
library access) based on administrative and behavioural indicators.

This project develops an AI-powered student-blocking prediction system using a feed-forward
**Deep Neural Network** built with TensorFlow/Keras.

The model analyzes six student indicators — attendance, grades, tuition payment status, late
arrivals, warnings, and grade level — and classifies each student into one of two categories:

- **Allowed:** Students who should keep normal access to school services.
- **Blocked:** Students who should be restricted due to poor attendance, unpaid tuition, or
  repeated academic/disciplinary issues.

## 2. Project Objectives

The main objectives of this project are:

1. To understand the application of Artificial Intelligence and Deep Learning in school
   administration.
2. To design and simulate a realistic student dataset.
3. To preprocess and normalize the data for a neural network.
4. To design and train a fully-connected neural network for binary classification.
5. To evaluate the model using standard classification metrics.
6. To analyze the model's strengths and limitations.

## 3. Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Model development and training |
| TensorFlow / Keras | Deep learning framework and model building |
| scikit-learn | Data splitting, scaling, and evaluation metrics |
| NumPy / Pandas | Data generation and manipulation |
| Matplotlib | Charts and visualizations |
| joblib | Model/scaler serialization (template script) |
| GitHub | Project documentation and source code hosting |

## 4. Dataset Description

A synthetic dataset of **3,000 students** was generated to simulate realistic school records.

| Feature | Description |
|---|---|
| `attendance_rate` | Attendance percentage (0–100) |
| `average_grade` | Average grade (0–20) |
| `tuition_paid` | Whether tuition fees are paid (0 = No, 1 = Yes) |
| `late_arrivals` | Number of late arrivals |
| `warnings_count` | Number of disciplinary warnings |
| `grade_level` | Student's grade level (1–6) |

The target label `is_blocked` was generated using a rule combining low attendance (< 75%),
unpaid tuition, and the combination of low grades with many absences.

### Dataset Distribution

| Category | Number of Students | Percentage |
|---|---:|---:|
| Allowed | 1,557 | 51.9% |
| Blocked | 1,443 | 48.1% |
| **Total** | **3,000** | **100%** |

![Data exploration](images/data_exploration.png)

### Dataset Splitting

The dataset was split into training and test sets using an 80:20 ratio, with stratification on
the target label.

| Dataset | Students | Percentage |
|---|---:|---:|
| Training | 2,400 | 80% |
| Test | 600 | 20% |
| **Total** | **3,000** | **100%** |

All features were standardized using `StandardScaler` before training.

## 5. Model Architecture

A fully-connected (dense) neural network was built with Keras.

### Model Layers

1. Input Layer (6 features)
2. Dense (32 units, ReLU activation)
3. Dropout (0.3)
4. Dense (16 units, ReLU activation)
5. Dropout (0.2)
6. Dense (1 unit, Sigmoid activation) — binary output

### Model Configuration

| Parameter | Value |
|---|---:|
| Input size | 6 features |
| Output | 1 (binary: Allowed / Blocked) |
| Optimizer | Adam |
| Loss function | Binary Crossentropy |
| Epochs | 30 |
| Batch size | 32 |
| Validation split | 20% of training data |
| Model format | Keras HDF5 (.h5) |

## 6. Model Training

The model was trained for 30 epochs, monitoring training/validation accuracy and loss.

![Training curves](images/training_curves.png)

### Final Recorded Training Results (Epoch 30)

| Metric | Result |
|---|---:|
| Training accuracy | 97.24% |
| Training loss | 0.0646 |
| Validation accuracy | 97.29% |
| Validation loss | 0.0723 |

The close gap between training and validation accuracy indicates the model generalizes well,
with no significant overfitting.

## 7. Model Evaluation

The model was evaluated on the held-out test set of 600 students.

### Overall Performance

**Test Accuracy: 96.33%**

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Allowed | 0.95 | 0.98 | 0.97 | 312 |
| Blocked | 0.97 | 0.95 | 0.96 | 288 |
| **Accuracy** | | | **0.96** | **600** |
| Macro Average | 0.96 | 0.96 | 0.96 | 600 |
| Weighted Average | 0.96 | 0.96 | 0.96 | 600 |

### Confusion Matrix

| Actual / Predicted | Allowed | Blocked |
|---|---:|---:|
| Allowed | 305 | 7 |
| Blocked | 15 | 273 |

![Confusion matrix](images/confusion_matrix.png)

### Performance Discussion

- **Allowed students:** Recall of 0.98 — the model correctly identifies almost all students who
  should keep normal access.
- **Blocked students:** Recall of 0.95 — the model correctly flags the large majority of
  students who should be restricted, with only 15 missed cases out of 288.
- Only 7 students were wrongly blocked (false positives) out of 600 test cases.

### Example Prediction

A sample student with low attendance (60%), average grade 10/20, paid tuition, 5 late arrivals
and 2 warnings was predicted with a blocking probability of **0.9999**, correctly classified as
**BLOCKED**.

## 8. Project Structure

```text
sigecedu-fraud-detection-miniproject/
│
├── README.md
├── main.py
├── sigecedu_fraud_detection.ipynb
│
└── images/
    ├── data_exploration.png
    ├── training_curves.png
    └── confusion_matrix.png
```

### File Descriptions

| File | Description |
|---|---|
| `sigecedu_fraud_detection.ipynb` | Main notebook: data generation, neural network, training, evaluation |
| `main.py` | Generic Random Forest template script (requires an external CSV dataset) |
| `README.md` | Project documentation |
| `images/` | Charts generated from the notebook (exploration, training, confusion matrix) |

## 9. Installation and Setup

### Prerequisites

- Python 3 compatible with the installed TensorFlow release.
- pip package installer.
- Jupyter Notebook or Google Colab.
- Git (if cloning the repository).

### Step 1: Clone the Repository

```bash
git clone https://github.com/danielkouame091-source/-sigecedu-fraud-detection-miniproject.git
cd -sigecedu-fraud-detection-miniproject
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

### Step 3: Run the Notebook

Open `sigecedu_fraud_detection.ipynb` in Jupyter or Google Colab and run all cells in order.
The dataset is generated automatically inside the notebook — no external CSV file is required.

The trained model is saved as `sigecedu_fraud_model.h5`, and the simulated dataset as
`students_data_simulated.csv`.

## 10. Results

- Dataset size: 3,000 simulated students.
- Number of classes: 2 (Allowed / Blocked).
- Training students: 2,400.
- Test students: 600.
- Final test accuracy: **96.33%**.
- Correctly classified test students: 578 out of 600.
- Model architecture: Dense Neural Network (32 → 16 → 1).

## 11. Limitations

1. The dataset is synthetically generated, not collected from a real school.
2. The blocking rule used to create labels is simplified compared to real administrative
   decisions.
3. The model has not been validated on real student records.
4. `main.py` is a generic template script and is not yet connected to the real dataset.

## 12. Future Improvements

1. Replace the simulated dataset with real, anonymized school records.
2. Compare the neural network against simpler models (Logistic Regression, Random Forest).
3. Add cross-validation to better estimate generalization performance.
4. Integrate the trained model into the SigecEdu school-management platform.
5. Add explainability (e.g. feature importance) to justify blocking decisions to staff.

## 13. Conclusion

This project demonstrates how a small deep neural network can learn to predict whether a
student's account should be blocked or allowed, based on simple administrative indicators.
Using a simulated dataset of 3,000 students, the model reached a test accuracy of **96.33%**,
with strong precision and recall on both classes. Future work should focus on validating the
approach on real school data before any production use.

## 14. References

1. TensorFlow Documentation: https://www.tensorflow.org/
2. Keras Documentation: https://keras.io/
3. scikit-learn Documentation: https://scikit-learn.org/
4. Google Colab: https://colab.research.google.com/

---

**Project Title:** SigecEdu Fraud/Blocking Detection Using Deep Learning

**Final Test Accuracy:** 96.33%

**Model Architecture:** Dense Neural Network (32 → 16 → 1)
