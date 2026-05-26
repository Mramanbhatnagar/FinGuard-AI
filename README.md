# FinGuard-AI
AI Intelligent Credit Risk &amp; Loan Underwriter
## Project Summary
FinGuard AI is a desktop application. It helps financial institutions assess commercial or personal loan risks. It uses a Scikit-Learn Random Forest Classifier. The classifier computes a quantitative credit risk score. Then, it sends the data to Google Gemini. Gemini generates a comprehensive underwriting report. The application saves the evaluation history into an SQLite database.
## Key Skills
Predictive AI: Local ML model execution (scikit-learn) for real-time risk classification.
Generative AI: Large Language Model integration (google-generativeai) for qualitative underwriting.
Production Engineering: Multi-threading, local data persistence (sqlite3), and clean OOP GUI design.
## Core Architecture
Inputs Collected: The CustomTkinter UI collects dynamic features (Income, Credit Score, Debt details).
Machine Learning Inference: The app loads a scikit-learn Random Forest Model (credit_model.pkl). It makes a prediction on risk.
Prompt Expansion: The numeric and predictive outputs combine into a structured string.
Generative Intelligence: This prompt goes to Google's Gemini. It translates numbers and predictions into a humanized underwriting report.
Database Storage: The data, predictions, and text reports are stored in SQLite for historical tracking.
