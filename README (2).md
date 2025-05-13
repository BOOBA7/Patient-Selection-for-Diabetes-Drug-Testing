# 🧬 Patient Selection for Diabetes Drug Testing — AI for Healthcare

This repository contains a complete data science project developed as part of the **"AI for Healthcare"** program by **Udacity**. The objective is to build a **regression model** that predicts patient hospitalization time, in order to **select suitable candidates** for a new diabetes drug clinical trial.

---

## 📚 Context

You are a data scientist at an innovative unicorn healthcare startup. Your company has developed a **revolutionary diabetes treatment**, which is about to enter **Phase III clinical trials**.

Due to the nature of this drug, patients must:

- Stay **5 to 7 days** in the hospital  
- Undergo **frequent monitoring and testing**  
- Receive **training via a mobile app** for medication adherence  

You have been provided with a **real-world patient dataset** and must identify patients likely to remain in the hospital long enough to qualify — without generating **excessive costs**.

🎯 **Goal**:  
Build a **regression model** to **predict hospitalization duration** and help **filter/select patients** for this testing phase.

---

## 🔍 Project Scope

### 1. 📊 Exploratory Data Analysis (EDA)
- Statistical profiling of key variables  
- Distribution analysis (e.g., age, medications, lab procedures)  
- Cardinality checks (e.g., `medical_specialty`, `diagnosis codes`)  
- Outlier detection and cleaning (e.g., removal of `"Unknown/Invalid"` genders)  

### 2. 🧹 Data Preprocessing
- Encoding categorical features  
- Normalization and transformation of skewed variables  
- Feature selection based on distribution and relevance  

### 3. 🧠 Predictive Modeling
- Supervised learning: **regression model**  
- Model evaluation and tuning  
- Interpretation of results to support **patient selection**

---

## 🧾 Dataset Highlights

Example features used:

| Feature                | Description                           |
|------------------------|---------------------------------------|
| `age`                 | Patient age bracket                   |
| `gender`              | Biological sex                        |
| `num_lab_procedures`  | Number of lab tests performed         |
| `num_medications`     | Total medications administered        |
| `number_inpatient`    | Past inpatient visits                 |
| `medical_specialty`   | Specialty of the treating physician   |
| `diagnosis codes`     | Primary and secondary diagnoses       |

---

## 🛠️ Tools & Technologies

- **Python**  
- **Jupyter Notebooks**  
- **pandas**, **NumPy**  
- **matplotlib**, **seaborn**  
- **scikit-learn** for preprocessing and modeling  

---

## 🎓 Learning Outcomes

This project demonstrates your ability to:

- Analyze and clean complex healthcare data  
- Engineer and encode clinical features  
- Train and evaluate regression models in a medical setting  
- Support **real-world clinical decisions** using AI  

---

## 💡 Notes

This project was designed, written, and documented with the assistance of **ChatGPT** by OpenAI to ensure clear explanations, code quality, and structured communication of insights.

---

## 👥 Author

**Anis Boubala**  
Veterinarian & Data Scientist in training — *AI for Healthcare, Udacity*  
Contact: anis.boubala [at] gmail.com  

Shared for educational review with:  
- **Laila Bellous** — Instructor  
- **Yannis Pandis** — Mentor  
