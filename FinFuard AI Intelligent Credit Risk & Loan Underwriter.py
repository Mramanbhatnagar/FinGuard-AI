import customtkinter as ctk
from tkinter import messagebox, filedialog
import sqlite3
import google.generativeai as genai
import threading
import pickle
import os
import numpy as np

# For the Predictive ML Engine (Used by AI Engineers)
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
API_KEY = "----API KEY----"  # Replace with a valid Gemini API Key
genai.configure(api_key=API_KEY)

DB_FILE = "loan_underwriting_v1.db"
MODEL_FILE = "credit_model.pkl"

# ==============================================================================
# 1. MACHINE LEARNING ENGINE SETUP (SIMULATED OR PRE-TRAINED)
# ==============================================================================
def init_ml_model():
    """Simulates/creates a quick ML model if it doesn't exist yet."""
    if not os.path.exists(MODEL_FILE):
        # Dummy data: [Income, Debt-to-Income %, Credit Score, Loan Amount (Thousands)]
        # 0 = Low Risk (Approve), 1 = High Risk (Reject)
        X = np.array([
            [85000, 15, 780, 50],
            [120000, 10, 810, 100],
            [35000, 45, 620, 20],
            [55000, 38, 580, 40],
            [95000, 22, 710, 80],
            [25000, 55, 520, 15]
        ])
        y = np.array([0, 0, 1, 1, 0, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)

init_ml_model()

# ==============================================================================
# 2. DATABASE SETUP
# ==============================================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT, income REAL, debt_ratio REAL,
            credit_score INTEGER, loan_amount REAL,
            ml_risk TEXT, ai_report TEXT, submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- MODERN UI SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# 3. FINGUARD APPLICATION CLASS
# ==============================================================================
class FinGuardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Config
        self.title("FinGuard AI | Intelligent Credit Underwriter")
        self.geometry("1100x720")

        # Layout Configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDEBAR (Controls) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="FinGuard AI 🛡️", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_assess = ctk.CTkButton(self.sidebar_frame, text="Evaluate Risk & Draft", command=self.start_evaluation_thread, fg_color="#2CC985", text_color="black")
        self.btn_assess.grid(row=1, column=0, padx=20, pady=10)

        self.btn_clear = ctk.CTkButton(self.sidebar_frame, text="Clear Entries", command=self.clear_fields, fg_color="#D35400")
        self.btn_clear.grid(row=2, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=5, column=0, padx=20, pady=20)

        # --- MAIN AREA (Inputs) ---
        self.scrollable_frame = ctk.CTkScrollableFrame(self, label_text="Applicant Financial Profile")
        self.scrollable_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Input Dictionary
        self.entries = {}
        
        self.fields = [
            ("Applicant Name", "Jane Doe"), ("Annual Income ($)", "75000"), 
            ("Debt-to-Income (%)", "20"), ("Credit Score (FICO)", "720"),
            ("Loan Amount ($)", "35000"), ("Employment Status", "Salaried - 5 yrs"),
            ("Down Payment ($)", "5000"), ("Collateral Value ($)", "0")
        ]

        # Create Grid of Inputs
        for i, (label_text, placeholder) in enumerate(self.fields):
            row = i // 2
            col = (i % 2) * 2
            
            lbl = ctk.CTkLabel(self.scrollable_frame, text=label_text, anchor="w")
            lbl.grid(row=row, column=col, padx=10, pady=(10, 0), sticky="w")
            
            entry = ctk.CTkEntry(self.scrollable_frame, placeholder_text=placeholder, width=250)
            entry.grid(row=row, column=col+1, padx=10, pady=(0, 10))
            self.entries[label_text] = entry

        # Additional Context / Explanatory Notes
        self.lbl_notes = ctk.CTkLabel(self.scrollable_frame, text="Applicant Explanatory Notes", anchor="w")
        self.lbl_notes.grid(row=len(self.fields)//2 + 1, column=0, padx=10, pady=(20, 0), sticky="w")
        
        self.txt_notes = ctk.CTkTextbox(self.scrollable_frame, height=80, width=600)
        self.txt_notes.grid(row=len(self.fields)//2 + 2, column=0, columnspan=4, padx=10, pady=(5, 20))

        # --- LOWER AREA (Results) ---
        self.result_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.result_frame.grid(row=1, column=1, sticky="nsew", padx=20, pady=(0, 20))
        
        self.lbl_result = ctk.CTkLabel(self.result_frame, text="AI Underwriting & Risk Report:", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_result.pack(anchor="w")
        
        self.result_box = ctk.CTkTextbox(self.result_frame, height=180, font=ctk.CTkFont(size=14))
        self.result_box.pack(fill="both", expand=True)

    def get_financial_data(self):
        data = {k: v.get() for k, v in self.entries.items()}
        data["Additional Notes"] = self.txt_notes.get("1.0", "end-1c")
        return data

    def start_evaluation_thread(self):
        self.status_label.configure(text="Status: Assessing...", text_color="#FFD700")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", "Computing ML model & consulting AI Underwriter... Please wait.")
        threading.Thread(target=self.evaluate_application).start()

    def evaluate_application(self):
        try:
            profile = self.get_financial_data()
            
            # --- PREDICTIVE ML ENGINE ---
            # Extract features for Local ML Model Prediction
            income = float(profile["Annual Income ($)"])
            dti = float(profile["Debt-to-Income (%)"])
            credit_score = int(profile["Credit Score (FICO)"])
            loan_amount = float(profile["Loan Amount ($)"])
            
            # Load ML model and execute inference
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            
            input_features = np.array([[income, dti, credit_score, loan_amount / 1000]])
            prediction = model.predict(input_features)[0]
            ml_risk_rating = "Low Risk (Auto-Approve Recommendation)" if prediction == 0 else "High Risk (Caution Suggested)"

            # --- GENERATIVE AI PROMPT ---
            prompt = (
                f"Act as a professional senior credit underwriter and draft a formal financial summary.\n"
                f"Evaluation Input Data: {profile}\n"
                f"Local Machine Learning Model Evaluation: {ml_risk_rating}\n\n"
                f"--- OUTPUT HEADINGS ---\n"
                f"Risk Analysis:\n"
                f"Local ML Rating:\n"
                f"Qualitative Assessment:\n"
                f"Mitigating Factors:\n"
                f"Underwriter Decision:\n"
                f"Terms & Conditions:\n"
                f"Recommended Next Steps:\n\n"
                f"Keep your underwriting tone professional and structured using bullet points."
            )

            # Gemini Inference
            ai_model = genai.GenerativeModel('gemini-1.5-flash')
            response = ai_model.generate_content(prompt)
            ai_report = response.text

            # Update UI safely
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", ai_report)
            self.status_label.configure(text="Status: Complete", text_color="#2CC985")
            
            # Save data to database
            self.save_to_db(profile["Applicant Name"], income, dti, credit_score, loan_amount, ml_risk_rating, ai_report)

        except Exception as e:
            self.result_box.delete("1.0", "end")
            self.result_box.insert("1.0", f"Error evaluating credit: {str(e)}")
            self.status_label.configure(text="Status: Error", text_color="red")

    def save_to_db(self, name, income, dti, credit, loan, ml_risk, ai_report):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            INSERT INTO applications (applicant_name, income, debt_ratio, credit_score, loan_amount, ml_risk, ai_report) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, income, dti, credit, loan, ml_risk, ai_report))
        conn.commit()
        conn.close()

    def clear_fields(self):
        for entry in self.entries.values():
            entry.delete(0, 'end')
        self.txt_notes.delete("1.0", "end")
        self.result_box.delete("1.0", "end")
        self.status_label.configure(text="Status: Ready", text_color="gray")

if __name__ == "__main__":
    app = FinGuardApp()
    app.mainloop()
