# QuantumScholar AI — AI Scholarship Recommendation & Eligibility System

**QuantumScholar AI** is a modern, full-stack educational technology platform powered by Streamlit, Python, Pandas, Openpyxl, Altair, and Google Gemini LLM. It helps students find government and private scholarships tailored to their academic background, degree level, community category, family income, and disability status.

---

## 🌟 Key Features

1. **Conversational First-Page Flow & Multi-Turn Memory**:
   - Natural language interface where students can describe their profile conversationally or via voice.
   - Built-in `CumulativeProfileManager` preserves and merges details across multiple dialogue turns.
   - Fast AI attribute extraction with Gemini NLP and rule-based fallback (< 1.0s).

2. **Smart Scoring & Recommendation Engine**:
   - Indexes and evaluates **72 verified government & private scholarship schemes** (Tamil Nadu State Welfare, Central Schemes, Reliance Foundation, NSP, etc.).
   - Strict qualification disambiguation:
     - Differentiates student disability schemes from schemes for children of differently abled parents.
     - Strict gender validation (e.g., Pudhumai Penn Thittam for female applicants).
     - Course & education level matching (UG vs. Pre-Matric school schemes).
     - Income ceiling checks & community quotas (BC / MBC / DNC / SC / ST / General).

3. **Continuous Voice Dictation**:
   - Outlined voice dictation button leveraging the browser Web Speech API.
   - Hands-free transcription directly parsed and merged into the active profile.

4. **Dedicated What-If Scenario Simulator**:
   - Interactive sliders and selectboxes to simulate grade boosts, income changes, or category shifts.
   - Real-time KPI comparisons (Base Top Score vs. Simulated Top Score with score delta).
   - Altair grouped bar chart visualizing scenario score fluctuations across candidate schemes.

5. **Profile Improvement Advisor**:
   - Actionable criteria tips targeting academic merit cutoffs (e.g., 60%+ for Reliance Foundation), 7.5% government school quota, and First Graduate verification.
   - Criteria Gap Analysis chart comparing the current profile against high-value scholarship benchmarks.

6. **Authentication & User Management**:
   - Secure sign-in and registration powered by salted SHA-256 password hashing.
   - Default demo credentials for instant evaluation.

---

## 🎨 UI & Design System

Built with a light SaaS theme:
- **Primary Background**: `#F7F8FA` (Ivory-white)
- **Deep Navy**: `#23364D` (Headings and navigation)
- **Primary Blue / Dusty Blue**: `#52769B` (Borders and progress indicators)
- **Primary Button Blue**: `#355E85` (Primary CTAs)
- **Cards**: `#FFFFFF` with subtle shadows and borders (`#E2E7ED`)
- **Status Badges**:
  - `✓ Eligible`: `#E2F3E8` (Text: `#246B45`)
  - `⚠ Potential Match`: `#FFF2D5` (Text: `#8A5A00`)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Shamini11/Scholarship-Recommendation.git
cd Scholarship-Recommendation
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key (Optional)
Set your Google Gemini API key as an environment variable (or use the built-in fallback):
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key"

# Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key"
```

### 5. Run the Application
```bash
streamlit run webapp.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Render

This repository includes `render.yaml`, `Procfile`, and `.streamlit/config.toml` configured for instant deployment on [Render](https://render.com).

### Quick Steps:
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repository: `Shamini11/Scholarship-Recommendation`.
4. Configure service settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run webapp.py --server.port $PORT --server.address 0.0.0.0`
5. *(Optional)* Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: *(Your Google Gemini API Key)*
6. Click **Deploy Web Service**. Render will build and deploy your app with a public `https://*.onrender.com` URL!

---

## 🧪 Testing

Run the automated unit test suite:
```bash
python test_suite.py
```
Covers:
- Dataset verification (72 schemes loaded)
- NLP attribute extraction
- Cumulative profile merge
- Strict disability and gender filtering
- URL verification & scoring bounds
- Authentication registration and login validation
- Voice transcript parsing
- What-If simulation and improvement advisor logic

---

## 📁 Project Structure

```
├── webapp.py                         # Streamlit web application & UI
├── engine.py                         # NLP extraction, cumulative profile, & scoring engine
├── auth.py                           # User authentication & password hashing
├── app.py                            # Dual-mode entry point (Web & CLI)
├── test_suite.py                     # Automated unit test suite
├── scholarship dataset 72.xlsx       # Primary scholarship dataset (72 verified schemes)
├── users.json                        # User database store
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── Scholarship_Research_Paper_LaTeX/ # Academic paper source & documentation
```
