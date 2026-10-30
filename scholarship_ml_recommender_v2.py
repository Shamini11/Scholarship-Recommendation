"""Tamil Nadu Scholarship Recommendation System - Version 2
TF-IDF + KNN + field-wise eligibility compatibility.
Unknown information is NEVER treated as No.
"""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA_CANDIDATES = [
    Path("Tamil_Nadu_Scholarship_Dataset_v1.xlsx"),
    Path("Tamil_Nadu_Scholarship_Dataset_v1(1).xlsx"),
]
SHEET_NAME = "TN_Scholarships"
OUTPUT_FILE = "recommendation_results_v2.csv"
TOP_K = 10

DATA_FILE = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_FILE is None:
    raise FileNotFoundError(
        "Excel dataset not found. Put Tamil_Nadu_Scholarship_Dataset_v1.xlsx "
        "in the same folder as this Python file."
    )

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

print("=" * 70)
print("SCHOLARSHIP RECOMMENDATION SYSTEM - VERSION 2")
print("=" * 70)
print(f"Dataset: {DATA_FILE.name}")
print(f"Scholarships loaded: {len(df)}")
print(f"Attributes preserved: {len(df.columns)}")

student_text = """I am a 20-year-old female student from Tamil Nadu studying second-year B.Tech Computer Science. I studied Classes 6 to 12 in a Government school. My family's annual income is around Rs. 1.8 lakh and I am the first graduate in my family. I need financial assistance for my studies."""

UNKNOWN = {"", "nan", "none", "not specified", "not applicable", "n/a", "na", "-"}

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm(x):
    return re.sub(r"\s+", " ", clean(x).lower()).strip()

def unknown(x):
    return norm(x) in UNKNOWN

def parse_number(x):
    s = norm(x).replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    n = float(m.group(1))
    if "lakh" in s:
        n *= 100000
    return n

def extract_profile(text):
    t = norm(text)
    p = {k: None for k in [
        "age","gender","state","education_level","course","stream","year",
        "income","government_school","first_graduate","community","minority",
        "disability","financial_need"
    ]}
    m = re.search(r"(\d{1,3})\s*[- ]?year[- ]?old", t)
    if m: p["age"] = int(m.group(1))
    if re.search(r"\bfemale\b|\bwoman\b|\bgirl\b", t): p["gender"] = "Female"
    elif re.search(r"\bmale\b|\bman\b|\bboy\b", t): p["gender"] = "Male"
    if "tamil nadu" in t: p["state"] = "Tamil Nadu"
    if re.search(r"b\.?\s*tech", t): p["course"] = "B.Tech"
    if re.search(r"computer science|cse", t): p["stream"] = "Computer Science"
    if re.search(r"undergraduate|\bug\b|bachelor|b\.?\s*tech", t): p["education_level"] = "UG"
    m = re.search(r"(first|1st|second|2nd|third|3rd|fourth|4th)\s*[- ]?year", t)
    if m:
        p["year"] = {"first":1,"1st":1,"second":2,"2nd":2,"third":3,"3rd":3,"fourth":4,"4th":4}[m.group(1)]
    m = re.search(r"(?:income|annual income).*?(?:rs\.?|₹)?\s*([\d,.]+)\s*(lakh|lakhs)?", t)
    if m:
        p["income"] = float(m.group(1).replace(",", "")) * (100000 if m.group(2) else 1)
    if re.search(r"government school|govt school|govt\. school", t): p["government_school"] = True
    if "first graduate" in t: p["first_graduate"] = True
    if re.search(r"financial assistance|financial need|need financial", t): p["financial_need"] = True
    for c in ["BC","MBC","DNC","SC","ST"]:
        if re.search(rf"\b{c.lower()}\b", t): p["community"] = c; break
    if "minority" in t: p["minority"] = True
    if re.search(r"disability|disabled", t): p["disability"] = True
    return p

profile = extract_profile(student_text)
print("\nEXTRACTED STUDENT PROFILE")
print("-" * 70)
for k,v in profile.items(): print(f"{k:20}: {v if v is not None else 'Unknown'}")

TEXT_COLUMNS = [
    "Scholarship_Name","Provider","Department","Scholarship_Type","State","Applicable_States",
    "Education_Level","Course_Category","Course_Name","Course_Stream","Year_of_Study",
    "Institution_Type","Institution_Requirement","Gender_Eligibility","Community_Eligibility",
    "Minority_Eligibility","Disability_Eligibility","Domicile_Requirement","Annual_Income_Limit",
    "Income_Type","Economic_Category","Financial_Need","Parent_Occupation","Sibling_Condition",
    "Government_School_Eligibility","First_Graduate_Eligibility","Rural_Urban_Eligibility",
    "Hosteller_Eligibility","Special_Category","Orphan_Status","Parent_Status","Benefit_Type",
    "Duration","Eligibility_Description","Keywords"
]

def scholarship_text(row):
    out=[]
    for c in TEXT_COLUMNS:
        if c in row.index and not unknown(row[c]):
            out.append(f"{c.replace('_',' ')} {clean(row[c])}")
    return " ".join(out)

df["ML_Text"] = df.apply(scholarship_text, axis=1)
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1, sublinear_tf=True)
X = vectorizer.fit_transform(df["ML_Text"])
knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
q = vectorizer.transform([student_text])
distances, indices = knn.kneighbors(q, n_neighbors=len(df))

def has_any(value, words):
    v = norm(value)
    return any(w in v for w in words)

def check(row, p):
    matched, conflicts, questions = [], [], []
    # State
    if p["state"]:
        s = " ".join(clean(row.get(c,"")) for c in ["State","Applicable_States","Domicile_Requirement"]).lower()
        if "tamil nadu" in s or "any state" in s or "india" in s or unknown(s):
            matched.append("Tamil Nadu/state compatibility")
        elif s:
            conflicts.append("State/domicile requirement may exclude Tamil Nadu")
    # Education/course
    if p["education_level"]:
        e = norm(row.get("Education_Level",""))
        if unknown(e) or has_any(e,["ug","undergraduate","degree","professional","any"]):
            matched.append("Undergraduate eligibility")
        elif has_any(e,["pg","postgraduate"]): conflicts.append("Scholarship is for postgraduate study")
    if p["course"]:
        c = " ".join(clean(row.get(x,"")) for x in ["Course_Category","Course_Name","Course_Stream","Eligibility_Description","Keywords"]).lower()
        if has_any(c,["b.tech","engineering","technical","professional","all courses","any course"]):
            matched.append("B.Tech/technical/professional compatibility")
        elif has_any(c,["class 9","class 10","class 11","class 12","school student"]):
            conflicts.append("Scholarship is school-level")
    # Gender
    if p["gender"]:
        g = norm(row.get("Gender_Eligibility",""))
        if unknown(g) or has_any(g,["any","all"]): pass
        elif p["gender"].lower() in g: matched.append("Gender eligibility")
        else: conflicts.append("Gender eligibility conflict")
    # Age
    if p["age"] is not None:
        amin,amax=parse_number(row.get("Age_Min","")),parse_number(row.get("Age_Max",""))
        if amin is not None and p["age"] < amin: conflicts.append("Below minimum age")
        elif amax is not None and p["age"] > amax: conflicts.append("Above maximum age")
        else: matched.append("Age compatibility")
    # Income
    if p["income"] is not None:
        lim=parse_number(row.get("Annual_Income_Limit",""))
        if lim is not None:
            if p["income"] <= lim: matched.append("Income within stated limit")
            else: conflicts.append("Income exceeds stated limit")
    # Govt school
    if p["government_school"] is True:
        v=norm(row.get("Government_School_Eligibility",""))
        if has_any(v,["yes","required","eligible","government school"]): matched.append("Government-school criterion")
        elif has_any(v,["no","not eligible"]): conflicts.append("Government-school criterion conflict")
    # First graduate
    if p["first_graduate"] is True:
        v=norm(row.get("First_Graduate_Eligibility",""))
        if has_any(v,["yes","required","eligible","first graduate"]): matched.append("First-graduate criterion")
        elif has_any(v,["no","not eligible"]): conflicts.append("First-graduate criterion conflict")
    # Community: unknown means ASK, never assume.
    cv=norm(row.get("Community_Eligibility",""))
    if p["community"]:
        if p["community"].lower() in cv or has_any(cv,["all","any"]): matched.append("Community eligibility")
        elif not unknown(cv): conflicts.append("Community eligibility may conflict")
    elif has_any(cv,["bc","mbc","dnc","sc","st"]):
        questions.append("What is the student's community/category?")
    # Minority
    mv=norm(row.get("Minority_Eligibility",""))
    if p["minority"] is None and has_any(mv,["yes","required","minority"]):
        questions.append("Does the student belong to a minority community?")
    # Disability / parent disability: critical for schemes like TN038.
    dv=" ".join(norm(row.get(c,"")) for c in ["Disability_Eligibility","Special_Category","Parent_Status"])
    if p["disability"] is None and has_any(dv,["disability","disabled","differently abled"]):
        questions.append("Does the student or parent/guardian meet the disability-related eligibility condition?")
    # Financial need
    if p["financial_need"]:
        fv=norm(row.get("Financial_Need",""))
        if has_any(fv,["yes","required","need-based","financial need"]): matched.append("Financial-need criterion")
    return matched, conflicts, list(dict.fromkeys(questions))

rows=[]
for distance,idx in zip(distances[0],indices[0]):
    row=df.iloc[idx]
    similarity=max(0.0,min(1.0,1-float(distance)))
    matched,conflicts,questions=check(row,profile)
    # Compatibility: only known checks contribute. Unknown gets zero credit.
    known=len(matched)+len(conflicts)
    compatibility=(len(matched)/known) if known else 0.50
    if conflicts: compatibility=min(compatibility,0.15)
    final=(0.40*similarity)+(0.60*compatibility)
    if conflicts: level="Low Match"
    elif final>=0.75: level="High Match - Verify" if questions else "High Match"
    elif final>=0.50: level="Good Match - Verify" if questions else "Good Match"
    else: level="Potential Match - More Information Needed" if questions else "Potential Match"
    rows.append({
        "Scholarship_ID":row.get("Scholarship_ID",""),
        "Scholarship_Name":row.get("Scholarship_Name",""),
        "ML_Similarity_Percent":round(similarity*100,2),
        "Eligibility_Compatibility_Percent":round(compatibility*100,2),
        "Final_Recommendation_Score_Percent":round(final*100,2),
        "Match_Level":level,
        "Matched_Reasons":"; ".join(matched) if matched else "No confirmed eligibility criterion yet.",
        "Missing_Information_Questions":"; ".join(questions) if questions else "None identified.",
        "Conflicts":"; ".join(conflicts) if conflicts else "None",
        "Benefit":row.get("Benefit_Type",""),
        "Scholarship_Amount":row.get("Scholarship_Amount",""),
        "Official_Source":row.get("Official_Source_URL",""),
        "Source_Verification_Status":row.get("Source_Verification_Status",""),
        "Application_Deadline":row.get("Application_Deadline","")
    })

results=pd.DataFrame(rows).sort_values("Final_Recommendation_Score_Percent",ascending=False).reset_index(drop=True)
results.insert(0,"Recommendation_Rank",range(1,len(results)+1))
results.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")

print("\n"+"="*70)
print("TOP 10 RECOMMENDATIONS")
print("="*70)
print(results[["Recommendation_Rank","Scholarship_ID","Scholarship_Name","ML_Similarity_Percent","Eligibility_Compatibility_Percent","Final_Recommendation_Score_Percent","Match_Level"]].head(TOP_K).to_string(index=False))
print("\n"+"="*70)
print(f"Saved: {OUTPUT_FILE}")
print("="*70)
print("Unknown information is NOT treated as No.")
print("Missing eligibility information is shown as a question.")
print("Scores are recommendation scores, NOT model accuracy.")
