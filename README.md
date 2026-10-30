# SigecEdu Fraud Detection Mini-Project

## Overview

This mini-project applies a Deep Learning model (a neural network built with TensorFlow/Keras)
to a school-management use case: automatically deciding whether a student's account should be
**blocked** or **allowed**, based on behavioural and administrative indicators such as attendance,
grades, and unpaid tuition fees.

## Abstract

Schools often need to flag students who should be temporarily restricted from certain services
(exam registration, library access, etc.) due to poor attendance, unpaid fees, or repeated
warnings. This project simulates a dataset of 3,000 students and trains a feed-forward neural
network to predict this blocking decision automatically, reaching **96.3% accuracy** on the test
set.

## Introduction

Manual, rule-based blocking decisions are slow and inconsistent across staff members. This
project explores whether a simple deep learning model can learn the underlying pattern from
historical data and reproduce (or improve on) the blocking decision automatically, which could
later be integrated into a school management system such as SigecEdu.

## Dataset

A synthetic dataset of 3,000 students was generated with the following features:

| Feature            | Description                                   |
|---------------------|------------------------------------------------|
| `attendance_rate`   | Attendance percentage (0–100)                  |
| `average_grade`     | Average grade (0–20)                           |
| `tuition_paid`      | Whether tuition fees are paid (0/1)            |
| `late_arrivals`     | Number of late arrivals                        |
| `warnings_count`    | Number of disciplinary warnings                |
| `grade_level`       | Student's grade level (1–6)                    |

The target label `is_blocked` is derived from a rule combining low attendance, unpaid tuition,
and the combination of low grades with many absences. 48.1% of simulated students are blocked.

![Data exploration](images/data_exploration.png)

## Model Architecture

A fully-connected neural network was built with Keras:

- Input layer (6 features)
- Dense(32, ReLU) → Dropout(0.3)
- Dense(16, ReLU) → Dropout(0.2)
- Dense(1, Sigmoid) — binary output (Blocked / Allowed)

Trained for 30 epochs with the Adam optimizer and binary cross-entropy loss.

![Training curves](images/training_curves.png)

## Results

| Metric      | Score |
|-------------|-------|
| Accuracy    | 96.3% |
| Precision (Blocked) | 0.97 |
| Recall (Blocked)    | 0.95 |

![Confusion matrix](images/confusion_matrix.png)

The model correctly classifies the large majority of students, with very few false blocks
(7 students wrongly blocked) and a small number of missed blocks (15 students wrongly allowed).

## How to Run

1. Open `sigecedu_fraud_detection.ipynb` in Jupyter or Google Colab.
2. Run all cells in order — the dataset is generated automatically (no external CSV needed).
3. The trained model is saved as `sigecedu_fraud_model.h5`, and the simulated data as
   `students_data_simulated.csv`.

## Project Structure

## Conclusion

This mini-project shows that a small neural network can learn a reliable blocking decision from
simple student indicators, reaching 96.3% accuracy. Future work could replace the simulated data
with real school records and compare the neural network against simpler models such as Logistic
Regression or Random Forest.
