"""
Tamil Nadu Scholarship Recommendation System
ML prototype using:
1. Rule-based eligibility checks for explicit constraints
2. TF-IDF text representation
3. K-Nearest Neighbors (cosine similarity) for recommendation ranking

Input:
    Tamil_Nadu_Scholarship_Dataset_v1(1).xlsx

Run:
    python scholarship_ml_recommender.py

Dependencies:
    pandas
    openpyxl
    scikit-learn
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# 1. FILE CONFIGURATION
# ------------------------------------------------------------

DATA_FILE = Path("Tamil_Nadu_Scholarship_Dataset_v1(1).xlsx")
SHEET_NAME = "TN_Scholarships"
TOP_K = 5


# ------------------------------------------------------------
# 2. DATA LOADING
# ------------------------------------------------------------

def load_dataset(path=DATA_FILE):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the Excel file in the same folder as this Python file."
        )

    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    required = [
        "Scholarship_ID",
        "Scholarship_Name",
        "Education_Level",
        "Course_Category",
        "Course_Name",
        "Course_Stream",
        "Gender_Eligibility",
        "Age_Min",
        "Age_Max",
        "Community_Eligibility",
        "Minority_Eligibility",
        "Disability_Eligibility",
        "Domicile_Requirement",
        "Annual_Income_Limit",
        "Government_School_Eligibility",
        "First_Graduate_Eligibility",
        "Eligibility_Description",
        "Keywords",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    return df.fillna("Not Specified")


# ------------------------------------------------------------
# 3. TEXT REPRESENTATION FOR TF-IDF
# ------------------------------------------------------------

# These attributes are most useful for semantic recommendation.
# The complete 52-column dataset is retained; only relevant
# eligibility/recommendation attributes are converted to ML text.

TEXT_COLUMNS = [
    "Scholarship_Name",
    "Scholarship_Type",
    "State",
    "Applicable_States",
    "Education_Level",
    "Course_Category",
    "Course_Name",
    "Course_Stream",
    "Year_of_Study",
    "Institution_Type",
    "Institution_Requirement",
    "Gender_Eligibility",
    "Community_Eligibility",
    "Minority_Eligibility",
    "Disability_Eligibility",
    "Domicile_Requirement",
    "Annual_Income_Limit",
    "Income_Type",
    "Economic_Category",
    "Financial_Need",
    "Parent_Occupation",
    "Sibling_Condition",
    "Government_School_Eligibility",
    "First_Graduate_Eligibility",
    "Rural_Urban_Eligibility",
    "Hosteller_Eligibility",
    "Special_Category",
    "Orphan_Status",
    "Parent_Status",
    "Benefit_Type",
    "Duration",
    "Eligibility_Description",
    "Keywords",
]


def clean_text(value):
    value = str(value)
    value = re.sub(r"[_/|,-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def build_scholarship_text(row):
    parts = []

    for col in TEXT_COLUMNS:
        if col in row.index:
            value = row[col]

            # Do not give the ML model much weight for missing placeholders.
            if str(value).strip().lower() in {
                "not specified",
                "not mentioned",
                "unknown",
                "n/a",
                "nan",
            }:
                continue

            # Repeating the name of an eligibility field gives TF-IDF
            # a little more context without changing the raw dataset.
            parts.append(f"{col} {value}")

    return clean_text(" ".join(parts))


# ------------------------------------------------------------
# 4. STUDENT PROFILE
# ------------------------------------------------------------

# This sample is the same type of student used in the website prototype.
# Change these values to test another student.

DEFAULT_STUDENT = {
    "age": 20,
    "gender": "female",
    "state": "Tamil Nadu",
    "education_level": "undergraduate",
    "course": "B.Tech Computer Science",
    "year": "2nd year",
    "income": 180000,
    "community": None,
    "minority": None,
    "disability": None,
    "government_school": True,
    "first_graduate": True,
    "rural_urban": None,
    "hosteller": None,
}


# ------------------------------------------------------------
# 5. SIMPLE NLP NORMALIZATION
# ------------------------------------------------------------

def normalize_student_text(text):
    """
    Lightweight NLP normalization.
    This is deliberately transparent for a college-project prototype.
    A future version can replace this with a stronger NLP model.
    """
    text = text.lower()

    replacements = {
        "b tech": "btech",
        "b.tech": "btech",
        "computer science engineering": "computer science",
        "computer science": "computer science",
        "tamilnadu": "tamil nadu",
        "girl": "female",
        "woman": "female",
        "boy": "male",
        "man": "male",
        "government school": "government school",
        "govt school": "government school",
        "first graduate": "first graduate",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def student_profile_to_text(profile):
    parts = [
        f"age {profile.get('age', '')}",
        f"gender {profile.get('gender', '')}",
        f"state {profile.get('state', '')}",
        f"education level {profile.get('education_level', '')}",
        f"course {profile.get('course', '')}",
        f"year {profile.get('year', '')}",
        f"annual family income {profile.get('income', '')}",
    ]

    if profile.get("community"):
        parts.append(f"community {profile['community']}")

    if profile.get("minority"):
        parts.append(f"minority {profile['minority']}")

    if profile.get("disability"):
        parts.append(f"disability {profile['disability']}")

    if profile.get("government_school") is not None:
        parts.append(
            "government school yes"
            if profile["government_school"]
            else "government school no"
        )

    if profile.get("first_graduate") is not None:
        parts.append(
            "first graduate yes"
            if profile["first_graduate"]
            else "first graduate no"
        )

    if profile.get("rural_urban"):
        parts.append(f"rural urban {profile['rural_urban']}")

    if profile.get("hosteller") is not None:
        parts.append(
            "hosteller yes" if profile["hosteller"] else "hosteller no"
        )

    return normalize_student_text(" ".join(parts))


# ------------------------------------------------------------
# 6. EXTRACT A FEW BASIC ATTRIBUTES FROM NATURAL LANGUAGE
# ------------------------------------------------------------

def extract_student_profile(text):
    """
    Transparent prototype NLP extractor.
    It is not presented as a final production NLP model.
    """

    original = text
    t = normalize_student_text(text)

    profile = {
        "age": None,
        "gender": None,
        "state": None,
        "education_level": None,
        "course": None,
        "year": None,
        "income": None,
        "community": None,
        "minority": None,
        "disability": None,
        "government_school": None,
        "first_graduate": None,
        "rural_urban": None,
        "hosteller": None,
    }

    # Age
    m = re.search(r"\b(\d{2})\s*(?:year|years)?\s*old\b", t)
    if m:
        profile["age"] = int(m.group(1))

    # Gender
    if re.search(r"\bfemale\b|\bwoman\b|\bgirl\b", t):
        profile["gender"] = "female"
    elif re.search(r"\bmale\b|\bman\b|\bboy\b", t):
        profile["gender"] = "male"

    # State
    if "tamil nadu" in t:
        profile["state"] = "tamil nadu"

    # Education
    if any(x in t for x in ["btech", "b.e", "bachelor", "undergraduate", "degree"]):
        profile["education_level"] = "undergraduate"
    elif any(x in t for x in ["postgraduate", "masters", "mtech", "m.e"]):
        profile["education_level"] = "postgraduate"

    # Course
    course_patterns = [
        r"\b(btech(?:\s+computer science)?)\b",
        r"\b(b\.tech(?:\s+computer science)?)\b",
        r"\b(computer science(?:\s+engineering)?)\b",
        r"\b(mtech(?:\s+computer science)?)\b",
    ]

    for pattern in course_patterns:
        m = re.search(pattern, t)
        if m:
            profile["course"] = m.group(1).replace("btech", "B.Tech")
            break

    # Year of study
    m = re.search(r"\b(first|1st|second|2nd|third|3rd|fourth|4th)\s+year\b", t)
    if m:
        profile["year"] = m.group(0)

    # Income examples: ₹1.8 lakh / 1.8 lakh / 180000
    m = re.search(
        r"(?:₹|rs\.?\s*)?(\d+(?:\.\d+)?)\s*(lakh|lakhs)",
        t,
    )
    if m:
        profile["income"] = float(m.group(1)) * 100000
    else:
        m = re.search(
            r"(?:income|family income|annual income)[^\d]{0,20}(\d{5,7})",
            t,
        )
        if m:
            profile["income"] = float(m.group(1))

    # Government school
    if "government school" in t:
        profile["government_school"] = True

    # First graduate
    if "not a first graduate" in t or "not first graduate" in t:
        profile["first_graduate"] = False
    elif "first graduate" in t:
        profile["first_graduate"] = True

    return profile


# ------------------------------------------------------------
# 7. ELIGIBILITY HELPERS
# ------------------------------------------------------------

def text_value(row, col):
    return str(row.get(col, "")).strip().lower()


def is_missing_value(value):
    return value in {
        "",
        "not specified",
        "not mentioned",
        "unknown",
        "n/a",
        "nan",
    }


def explicit_gender_conflict(row, gender):
    if not gender:
        return False

    allowed = text_value(row, "Gender_Eligibility")

    if is_missing_value(allowed) or allowed == "any":
        return False

    if gender == "female" and "male" in allowed and "female" not in allowed:
        return True

    if gender == "male" and "female" in allowed and "male" not in allowed:
        return True

    return False


def explicit_school_conflict(row, government_school):
    if government_school is None:
        return False

    rule = text_value(row, "Government_School_Eligibility")

    if is_missing_value(rule):
        return False

    if government_school and ("not eligible" in rule or "not required" in rule):
        return False

    return False


def income_limit_value(value):
    """
    Extract a rough numeric annual-income ceiling.
    Supports values such as:
    '₹2.5 lakh', '2.5 lakhs', '250000'.
    """
    value = str(value).lower().replace(",", "")

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs)", value)
    if m:
        return float(m.group(1)) * 100000

    m = re.search(r"\b(\d{5,8})\b", value)
    if m:
        return float(m.group(1))

    return None


def calculate_eligibility_penalty(row, profile):
    """
    Returns:
        penalty: 0 to 1
        reasons: explanations
        conflicts: clear contradictions
    """

    penalty = 0.0
    reasons = []
    conflicts = []

    # Gender
    if explicit_gender_conflict(row, profile.get("gender")):
        penalty += 0.60
        conflicts.append("Gender requirement does not match the student.")

    # State / domicile
    state_rule = (
        text_value(row, "State")
        + " "
        + text_value(row, "Applicable_States")
        + " "
        + text_value(row, "Domicile_Requirement")
    )

    student_state = str(profile.get("state") or "").lower()

    if student_state == "tamil nadu" and state_rule:
        if "tamil" in state_rule and "nadu" in state_rule:
            reasons.append("Tamil Nadu requirement matches.")

    # Income
    income = profile.get("income")
    limit = income_limit_value(row.get("Annual_Income_Limit", ""))

    if income is not None and limit is not None:
        if income > limit:
            penalty += 0.60
            conflicts.append("Student income appears to exceed the stated limit.")
        else:
            reasons.append("Student income is within the stated income limit.")

    # Government school
    school_rule = text_value(row, "Government_School_Eligibility")

    if profile.get("government_school") is True and "required" in school_rule:
        reasons.append("Government-school condition matches.")

    # First graduate
    fg_rule = text_value(row, "First_Graduate_Eligibility")

    if profile.get("first_graduate") is True and "required" in fg_rule:
        reasons.append("First-graduate condition matches.")

    return min(penalty, 1.0), reasons, conflicts


# ------------------------------------------------------------
# 8. TF-IDF + KNN MODEL
# ------------------------------------------------------------

class ScholarshipRecommender:

    def __init__(self, df):
        self.df = df.copy()

        self.df["ML_Text"] = self.df.apply(
            build_scholarship_text,
            axis=1,
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )

        self.X = self.vectorizer.fit_transform(self.df["ML_Text"])

        self.knn = NearestNeighbors(
            n_neighbors=min(TOP_K, len(self.df)),
            metric="cosine",
            algorithm="brute",
        )

        self.knn.fit(self.X)

    def recommend(self, student_text, student_profile, top_k=TOP_K):
        student_text = normalize_student_text(student_text)

        student_vector = self.vectorizer.transform([student_text])

        distances, indices = self.knn.kneighbors(
            student_vector,
            n_neighbors=min(top_k, len(self.df)),
        )

        results = []

        for distance, index in zip(distances[0], indices[0]):
            row = self.df.iloc[index]

            similarity = max(0.0, 1.0 - float(distance))

            penalty, reasons, conflicts = calculate_eligibility_penalty(
                row,
                student_profile,
            )

            # Hybrid score:
            # 40% semantic similarity + 60% eligibility compatibility.
            eligibility_score = max(0.0, 1.0 - penalty)

            final_score = (
                0.40 * similarity
                + 0.60 * eligibility_score
            )

            if conflicts:
                match_level = "Low Match"
            elif final_score >= 0.75:
                match_level = "High Match"
            elif final_score >= 0.50:
                match_level = "Good Match"
            else:
                match_level = "Potential Match"

            results.append({
                "Scholarship_ID": row["Scholarship_ID"],
                "Scholarship_Name": row["Scholarship_Name"],
                "ML_Similarity": round(similarity * 100, 2),
                "Eligibility_Compatibility": round(
                    eligibility_score * 100, 2
                ),
                "Final_Recommendation_Score": round(
                    final_score * 100, 2
                ),
                "Match_Level": match_level,
                "Matched_Reasons": reasons,
                "Conflicts": conflicts,
                "Benefit": row.get("Benefit_Type", ""),
                "Scholarship_Amount": row.get("Scholarship_Amount", ""),
                "Official_Source": row.get("Official_Source_URL", ""),
            })

        result_df = pd.DataFrame(results)

        # Do not allow a clearly conflicting scholarship to rank above
        # non-conflicting scholarships merely because of word similarity.
        result_df["_conflict"] = result_df["Conflicts"].apply(bool)

        result_df = result_df.sort_values(
            by=["_conflict", "Final_Recommendation_Score"],
            ascending=[True, False],
        ).drop(columns="_conflict")

        return result_df.reset_index(drop=True)


# ------------------------------------------------------------
# 9. DISPLAY
# ------------------------------------------------------------

def print_profile(profile):
    print("\n" + "=" * 70)
    print("EXTRACTED STUDENT PROFILE")
    print("=" * 70)

    labels = {
        "age": "Age",
        "gender": "Gender",
        "state": "State",
        "education_level": "Education Level",
        "course": "Course",
        "year": "Year of Study",
        "income": "Annual Family Income",
        "community": "Community",
        "minority": "Minority Status",
        "disability": "Disability Status",
        "government_school": "Government School",
        "first_graduate": "First Graduate",
    }

    for key, label in labels.items():
        value = profile.get(key)

        if value is None:
            value = "Not mentioned"

        if key == "income" and isinstance(value, (int, float)):
            value = f"₹{value:,.0f}"

        print(f"{label:<25}: {value}")


def print_results(results):
    print("\n" + "=" * 70)
    print("TOP SCHOLARSHIP RECOMMENDATIONS")
    print("=" * 70)

    for i, row in results.iterrows():
        print(f"\n#{i + 1}  {row['Scholarship_Name']}")
        print(f"    ID                         : {row['Scholarship_ID']}")
        print(f"    Match Level                : {row['Match_Level']}")
        print(f"    ML Similarity              : {row['ML_Similarity']}%")
        print(
            "    Eligibility Compatibility  : "
            f"{row['Eligibility_Compatibility']}%"
        )
        print(
            "    Final Recommendation Score : "
            f"{row['Final_Recommendation_Score']}%"
        )

        if row["Matched_Reasons"]:
            print("    Matched / Supporting Criteria:")
            for reason in row["Matched_Reasons"]:
                print(f"      ✓ {reason}")

        if row["Conflicts"]:
            print("    Conflicts:")
            for conflict in row["Conflicts"]:
                print(f"      ✗ {conflict}")

        print(f"    Benefit                    : {row['Benefit']}")
        print(f"    Amount                     : {row['Scholarship_Amount']}")


# ------------------------------------------------------------
# 10. MAIN DEMO
# ------------------------------------------------------------

def main():
    print("\nTamil Nadu Scholarship Recommendation System")
    print("Hybrid NLP + TF-IDF + KNN ML Prototype")

    df = load_dataset()

    print(f"\nDataset loaded successfully: {len(df)} scholarships")
    print(f"Number of attributes: {len(df.columns)}")

    model = ScholarshipRecommender(df)

    student_text = (
        "I am a 20-year-old female student from Tamil Nadu "
        "studying second-year B.Tech Computer Science. "
        "I studied Classes 6 to 12 in a Government school. "
        "My family's annual income is around ₹1.8 lakh "
        "and I am the first graduate in my family."
    )

    print("\n" + "=" * 70)
    print("STUDENT INPUT")
    print("=" * 70)
    print(student_text)

    profile = extract_student_profile(student_text)
    print_profile(profile)

    student_ml_text = student_profile_to_text(profile)

    results = model.recommend(
        student_ml_text,
        profile,
        top_k=TOP_K,
    )

    print_results(results)

    # Save results for the website/backend integration later.
    results.to_csv(
        "recommendation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("Saved: recommendation_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
