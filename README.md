# 📊 Customer Churn Prediction – Streamlit Web App

An **end‑to‑end Machine Learning web application** that predicts whether a telecom customer is likely to churn based on provided demographic & service‑related details.

🔗 **Live App:** [https://customer-churn-raj.streamlit.app/](https://customer-churn-raj.streamlit.app/)

📁 **Repository:** [https://github.com/rajlaljipandey/Customer-Churn-AI-WebApp]

---

## 🚀 Features

✔ Predicts whether a customer will churn or stay using ML model (Logistic Regression / Random Forest)
✔ Interactive form UI using Streamlit sidebar
✔ Shows **churn probability** with visual progress gauge
✔ Allows **PDF download** of the report & prediction output
✔ Deployed on **Streamlit Cloud**
✔ Fully open‑source – modify & use in portfolio / resume

---

## 🧠 Model Overview

The model is trained using **Telco Customer Churn dataset (public dataset)** which includes:

* 📌 Demographics (gender, senior citizen, dependents, partner)
* 📌 Services (phone service, paperless billing, contract type, etc.)
* 📌 Tenure in months

The model predicts:

```
1 → Customer is likely to churn
0 → Customer will stay
```

Additionally, a churn probability score (0‑100%) is displayed.

---

## 🛠️ Tech Stack

| Component        | Technology      |
| ---------------- | --------------- |
| Web Framework    | Streamlit       |
| ML Model         | Scikit‑Learn    |
| Backend Language | Python          |
| Deployment       | Streamlit Cloud |
| PDF Report       | fpdf‑python     |

---

## 🧰 Installation – Run Locally

### 1️⃣ Clone Repo

```bash
git clone <your_repo_url>
cd Customer-Churn-AI-WebApp
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Input Example

Users enter:

```yaml
Gender: Male/Female
Senior Citizen: 0/1
Partner: Yes/No
Dependents: Yes/No
Tenure Months: slider input
Phone Service: Yes/No
Paperless Billing: Yes/No
```

App returns:

```
🟢 Safe: Customer unlikely to churn
or
🔴 High Risk: Customer likely to churn
```

with a probability score.

---

## 📸 Screenshots

### 🧾 Input Form
![Input Form](screenshots/app-home.PNG)

### 📊 Prediction Output
![Prediction Result](screenshots/app-result.PNG)


---

## ☁ Deployment Notes

App deployed using **Streamlit Cloud**.
Push to GitHub → Streamlit auto‑deploys.
If dependency errors occur:

* Ensure **requirements.txt** includes `fpdf`, `scikit‑learn`, `streamlit`

---

## 🧑‍💻 Author

**Raj Lalji Pandey**
Portfolio‑style ML & Streamlit applications – ideal for Data Analyst / ML Engineer roles

🌐 GitHub: [https://github.com/rajlaljipandey](https://github.com/rajlaljipandey)
🔗 LinkedIn: [https://www.linkedin.com/in/raj-pandey-51288a237/](https://www.linkedin.com/in/raj-pandey-51288a237/)

---

## ⭐ Contribute / Support

Want to add improvements?
Pull requests are welcome!

```bash
# Create branch
git checkout -b new-feature
# Commit
git commit -m "added feature"
# Push
git push origin new-feature
```

Give ⭐ on GitHub if you found it useful 🙌
