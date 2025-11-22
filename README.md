# AI based Early Sepsis Detection and Prediction System

This project focuses on building a real-time sepsis prediction system using clinical time-series data. The goal is to detect early signs of sepsis before clinical diagnosis, enabling timely intervention and improved patient outcomes.

## 📌 Project Overview

Sepsis is a life-threatening condition caused by the body's extreme response to infection. Early prediction is challenging but critical.
In this project, we:

Clean and preprocess a large clinical dataset

Engineer temporal features for model learning

Implement multiple ML & DL models

Evaluate performance using AUROC, AUPRC, F1-Score, Precision & Recall

Prepare the framework for deployment-ready pipelines

## 📁 Dataset

We use an openly available clinical dataset containing over 1.5 million hourly records with 40+ physiological variables.

Key variables include:
Heart Rate, O2 Saturation, Temperature, Blood Pressure, Respiratory Rate, Creatinine, Bilirubin, Lactate, Platelets, etc.

The dataset also provides:

SepsisLabel (binary supervision)

Patient_ID (for time-based grouping)

ICU LOS (ICULOS)

Demographics

## 🔧 Workflow
1. Data Preprocessing

Handle missing values

Group by patient and perform forward/backward fill

Normalize and standardize features

Split dataset into train/validation/test sets

Address strong class imbalance

2. Feature Engineering

Temporal trend features (difference values, rolling features)

Physiological score-related variables

Model-ready input vectors

3. Model Development

We plan to implement and compare:

XGBoost Classifier (baseline)

LSTM-based Deep Learning Model

GRU / BiLSTM variations

1D-CNN for time-series

Hybrid fusion models (optional)

Each model is evaluated for early sepsis detection capability.

4. Evaluation Metrics

AUROC

AUPRC

F1 Score

Precision–Recall Curve

Sensitivity at low FPR

Optimal threshold search

## 📌 Current Progress

Dataset analysis completed

Preprocessing & temporal feature engineering done

Baseline XGBoost model implemented

Deep learning model architecture defined

Hyperparameter search in progress

## 🚀 Upcoming Work

Full DL model training (LSTM, GRU, CNN)

Model comparison and result documentation

Creation of inference pipeline

Deployment plan (FastAPI / Streamlit)

## 👥 Team Members

Priyanshu Thakur

Anmol Nayyar

Zorawar Singh Sandhu

Tanu Shree

## 📜 License

This project is for academic and research purposes only.
