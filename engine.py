"""
engine.py - Core Engine for AI Scholarship Recommendation & Eligibility System
Contains:
1. Data loading and preprocessing from 'scholarship dataset 72.xlsx'
2. Natural Language Profile Extraction using Google Gemini (gemini-3.8-flash) with robust NLP fallback
3. Cumulative Profile State Management
4. Smart Scoring & Matching Engine with strict disability & gender logic
"""

import os
import re
import json
import time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Default API key provided via environment variable
DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MODEL_NAME = "gemini-3.8-flash"

# Profile Schema Template
EMPTY_PROFILE = {
    "state": None,
    "community": None,
    "education_level": None,
    "annual_income": None,
    "gender": None,
    "disability": None,
    "min_marks_or_cgpa": None,
    "government_school": None,
    "first_graduate": None,
    "hosteller": None
}


class ScholarshipDataLoader:
    """Loads and preprocesses the scholarship dataset."""
    def __init__(self, excel_path: str = "scholarship dataset 72.xlsx", sheet_name: str = "Scholarships_Actual"):
        self.excel_path = excel_path
        self.sheet_name = sheet_name
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.excel_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, self.excel_path)
            if os.path.exists(candidate):
                self.excel_path = candidate
            else:
                raise FileNotFoundError(f"Scholarship dataset file not found at: {self.excel_path}")
        
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        
        # Pre-clean string columns to avoid float/None errors
        string_cols = [
            'Scholarship_Name', 'Provider', 'Scholarship_Amount', 'State',
            'Applicable_States', 'Community_Eligibility', 'Education_Level',
            'Annual_Income_Limit', 'Gender_Eligibility', 'Disability_Eligibility',
            'Government_School_Eligibility', 'First_Graduate_Eligibility',
            'Hosteller_Eligibility', 'Minimum_Marks', 'Minimum_CGPA',
            'Official_Source_URL', 'Eligibility_Description', 'Keywords',
            'Documents_Required', 'Application_Portal'
        ]
        
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.strip()
            else:
                df[col] = ''
                
        # Parse numeric income limit if present for fast vector indexing
        df['Parsed_Income_Limit'] = df['Annual_Income_Limit'].apply(self._parse_income_number)
        
        return df

    @staticmethod
    def _parse_income_number(val: str) -> Optional[float]:
        if not val or any(x in val.lower() for x in ['no income', 'not specified', 'no economic', 'nan']):
            return None
        # Extract digits
        clean_val = val.replace('₹', '').replace(',', '').replace(' ', '')
        matches = re.findall(r'(\d+(?:\.\d+)?)', clean_val)
        if matches:
            try:
                num = float(matches[0])
                if 'lakh' in val.lower() and num < 100:
                    num *= 100000
                return num
            except:
                return None
        return None


class RuleBasedNLPExtractor:
    """High-accuracy fallback NLP extractor using regex and semantic entity matching."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, Any]:
        extracted = {}
        t = text.lower()
        
        # 1. State
        if re.search(r'\b(tamil nadu|tamilnadu|tn)\b', t):
            extracted['state'] = "Tamil Nadu"
        elif re.search(r'\b(kerala|karnatka|karnataka|andhra|telangana|delhi|maharashtra|jammu|kashmir|punjab|bihar|up|west bengal)\b', t):
            m = re.search(r'\b(kerala|karnatka|karnataka|andhra|telangana|delhi|maharashtra|jammu|kashmir|punjab|bihar|up|west bengal)\b', t)
            extracted['state'] = m.group(1).title()
        elif re.search(r'\b(all india|india|pan india|national)\b', t):
            extracted['state'] = "All India"

        # 2. Gender
        if re.search(r'\b(female|woman|women|girl|daughter|she|her)\b', t):
            extracted['gender'] = "Female"
        elif re.search(r'\b(male|man|men|boy|son|he|his)\b', t):
            extracted['gender'] = "Male"
        elif re.search(r'\b(transgender|third gender)\b', t):
            extracted['gender'] = "Transgender"

        # 3. Community
        if re.search(r'\b(mbc|most backward)\b', t):
            extracted['community'] = "MBC"
        elif re.search(r'\b(dnc|denotified)\b', t):
            extracted['community'] = "DNC"
        elif re.search(r'\b(bc|backward class|obc)\b', t):
            extracted['community'] = "BC"
        elif re.search(r'\b(sc|scheduled caste|adi dravidar)\b', t):
            extracted['community'] = "SC"
        elif re.search(r'\b(st|scheduled tribe|tribal)\b', t):
            extracted['community'] = "ST"
        elif re.search(r'\b(oc|fc|general|forward caste|open)\b', t):
            extracted['community'] = "OC"
        elif re.search(r'\b(minority|muslim|christian|sikh|jain|buddhist)\b', t):
            extracted['community'] = "Minority"

        # 4. Education Level
        t_no_school_type = re.sub(r'\b(govt|government|private|matriculation|cbse)\s+schools?\b', '', t)
        if re.search(r'\b(ug|undergraduate|b\.?tech|b\.?e|b\.?sc|b\.?com|b\.?a|mbbs|bds|bachelor|college)\b', t):
            extracted['education_level'] = "UG"
        elif re.search(r'\b(pg|postgraduate|m\.?tech|m\.?e|m\.?sc|m\.?com|mba|mca|master)\b', t):
            extracted['education_level'] = "PG"
        elif re.search(r'\b(diploma|polytechnic)\b', t):
            extracted['education_level'] = "Diploma"
        elif re.search(r'\b(iti)\b', t):
            extracted['education_level'] = "ITI"
        elif re.search(r'\b(phd|doctorate|research scholar|fellowship)\b', t):
            extracted['education_level'] = "PhD"
        elif re.search(r'\b(10th|12th|plus two|\+2|class 10|class 12|pre-matric|in school|studying school|studying in school)\b', t) or re.search(r'\bschool\b', t_no_school_type):
            extracted['education_level'] = "School"

        # 5. Annual Income
        # Check patterns like "2.5 lakhs", "250000", "1 lakh", "income of 2,00,000"
        inc_match = re.search(r'(?:income\s*(?:of|is|:|=)?\s*(?:rs\.?|inr|₹)?\s*|annual\s*income\s*(?:of|is|:|=)?\s*|\b)(\d+(?:\.\d+)?)\s*(lakhs?|lac|l\b)', t)
        if inc_match:
            try:
                num = float(inc_match.group(1))
                extracted['annual_income'] = num * 100000.0
            except:
                pass
        else:
            inc_exact = re.search(r'(?:rs\.?|inr|₹)\s*(\d{1,3}(?:,\d{2,3})+|\d{4,7})', t)
            if not inc_exact:
                inc_exact = re.search(r'\b(\d{5,7})\b', t)
            if inc_exact:
                try:
                    num_str = inc_exact.group(1).replace(',', '')
                    extracted['annual_income'] = float(num_str)
                except:
                    pass

        # 6. Disability (Distinguishing student vs parent)
        if re.search(r'\b(no disability|not disabled|normal|none|nil|no handicap)\b', t):
            extracted['disability'] = "None"
        elif re.search(r'\b(parent|father|mother|dad|mom)\b.*\b(disab|differently abled|handicap|pwd)\b', t) or \
             re.search(r'\b(child|son|daughter)\s+of\s+(?:a\s+)?(?:differently abled|disabled)\b', t):
            extracted['disability'] = "Parent is differently abled"
        elif re.search(r'\b(walking disability|locomotor|orthopedic|physically handicapped|pwd|differently abled|blind|visually impaired|deaf|hearing impaired|speech impaired|paralysis|wheelchair|disability|disabled)\b', t):
            m = re.search(r'\b(walking disability|locomotor disability|locomotor|orthopedic|physically handicapped|pwd|differently abled|blind|visually impaired|deaf|hearing impaired|speech impaired|wheelchair|disability)\b', t)
            matched_dis = m.group(1).title() if m else "Locomotor"
            extracted['disability'] = matched_dis

        # 7. Marks or CGPA
        # e.g., "85%", "85 marks", "8.5 cgpa", "9.1 gpa"
        pct_match = re.search(r'(\d{2}(?:\.\d+)?)\s*(?:%|percent|percentage|marks)', t)
        if pct_match:
            try:
                extracted['min_marks_or_cgpa'] = float(pct_match.group(1))
            except:
                pass
        else:
            cgpa_match = re.search(r'(\d(?:\.\d+)?)\s*(?:cgpa|gpa|pointer)', t)
            if cgpa_match:
                try:
                    extracted['min_marks_or_cgpa'] = float(cgpa_match.group(1))
                except:
                    pass

        # 8. Government School Eligibility
        if re.search(r'\b(govt school|government school|studied in govt|6th to 12th in govt|govt\. school)\b', t):
            extracted['government_school'] = "Yes"
        elif re.search(r'\b(private school|matriculation school|cbse school|icse)\b', t):
            extracted['government_school'] = "No"

        # 9. First Graduate Eligibility
        if re.search(r'\b(first graduate|first generation graduate|first in family to (?:graduate|attend college))\b', t):
            extracted['first_graduate'] = "Yes"
        elif re.search(r'\b(not first graduate|parents are graduates)\b', t):
            extracted['first_graduate'] = "No"

        # 10. Hosteller Eligibility
        if re.search(r'\b(hostellers?|hostel students?|staying in hostel|living in hostel|hostel accommodation)\b', t):
            extracted['hosteller'] = "Hosteller"
        elif re.search(r'\b(day scholars?|dayscholars?|commuters?|staying at home|day students?)\b', t):
            extracted['hosteller'] = "Day Scholar"

        return extracted


class GeminiNLPExtractor:
    """Uses Google Gemini LLM (gemini-3.8-flash) to extract structured applicant profile."""
    
    def __init__(self, api_key: str = DEFAULT_GEMINI_API_KEY, model_name: str = DEFAULT_MODEL_NAME):
        self.api_key = api_key
        self.model_name = model_name
        self.last_status = "Initialized"

    def extract_with_gemini(self, text: str, current_profile: Dict[str, Any], user_name: str = "Student", timeout_sec: float = 1.2) -> Tuple[Dict[str, Any], str, str]:
        """
        Attempts extraction with Gemini gemini-3.8-flash acting as an AI Agent.
        Uses a daemon worker thread with a strict timeout so prediction is never delayed (> 1.2s).
        Returns (extracted_dict, source_info, ai_reply).
        """
        import threading

        gemini_result = []

        def _worker():
            try:
                from google import genai
                from google.genai import types
                
                client = genai.Client(api_key=self.api_key)
                prompt = f"""
You are an intelligent, empathetic, and professional Scholarship AI Advisor (acting like Gemini or ChatGPT) for Indian and Tamil Nadu scholarships.
The applicant's name is '{user_name}'.

The applicant just sent this message:
"{text}"

Current known profile:
{json.dumps(current_profile)}

Task:
1. Provide a warm, conversational, and helpful 'ai_reply' addressing the user by name where appropriate. Acknowledge what details they shared, explain what you understand, and gently suggest if any helpful info (like family income, government school, or first graduate status) might unlock more scholarships.
2. Extract or update the structured applicant profile as 'extracted_profile' with keys:
  "state": string or null (e.g., "Tamil Nadu", "All India")
  "community": string or null (e.g., "BC", "MBC", "DNC", "SC", "ST", "OC", "General", "Minority")
  "education_level": string or null (e.g., "UG", "PG", "Diploma", "ITI", "School", "PhD")
  "annual_income": float in INR or null (e.g., 250000.0 if user mentions 2.5 lakhs)
  "gender": string or null ("Female", "Male", "Transgender")
  "disability": string or null (e.g., "Walking Disability", "Locomotor", "Visual", "Parent is differently abled", "None")
  "min_marks_or_cgpa": float or null (e.g., 85.0 or 8.5)
  "government_school": string or null ("Yes" or "No")
  "first_graduate": string or null ("Yes" or "No")
  "hosteller": string or null ("Hosteller" or "Day Scholar")

Return ONLY a valid JSON dictionary:
{{
  "ai_reply": "...",
  "extracted_profile": {{ ... }}
}}
"""
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)
                parsed = json.loads(raw)
                gemini_result.append(parsed)
            except Exception:
                pass

        # Run worker as a daemon thread
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)

        if gemini_result:
            data = gemini_result[0]
            self.last_status = f"Gemini AI ({self.model_name})"
            extracted = data.get("extracted_profile", data)
            ai_reply = data.get("ai_reply", "")
            if not ai_reply:
                ai_reply = self._build_fallback_reply(extracted, current_profile, user_name)
            return extracted, self.last_status, ai_reply
        else:
            # Ultra-fast semantic NLP engine (< 5 milliseconds)
            self.last_status = "Scholarship AI Agent"
            rule_extracted = RuleBasedNLPExtractor.extract(text)
            ai_reply = self._build_fallback_reply(rule_extracted, current_profile, user_name)
            return rule_extracted, self.last_status, ai_reply

    def _build_fallback_reply(self, extracted: Dict[str, Any], current_profile: Dict[str, Any], user_name: str = "Student") -> str:
        """Constructs an empathetic, smart conversational response as an AI agent."""
        merged, changed = CumulativeProfileManager.merge(current_profile, extracted)
        name_greet = f"Hi {user_name}!" if user_name and user_name != "Student" else "Hello!"
        
        if not changed and not any(v for v in merged.values() if v is not None):
            return (
                f"{name_greet} I'm your Scholarship AI Advisor. Please tell me a little about yourself — "
                "such as your course of study (UG/PG), caste category (BC/MBC/SC/ST/General), marks, "
                "family income, or state. You can either type your message or click the 🎙️ mic icon to speak!"
            )
        
        details = []
        if merged.get('education_level'):
            details.append(f"pursuing **{merged['education_level']}**")
        if merged.get('gender'):
            details.append(f"**{merged['gender']}**")
        if merged.get('community'):
            details.append(f"**{merged['community']}** category")
        if merged.get('state'):
            details.append(f"in **{merged['state']}**")
        if merged.get('min_marks_or_cgpa'):
            m = merged['min_marks_or_cgpa']
            details.append(f"with **{m}%** marks" if m > 10 else f"with **{m}** CGPA")
        if merged.get('disability') and merged['disability'] not in ['None', 'no', 'null']:
            details.append(f"with **{merged['disability']}**")
        if merged.get('annual_income'):
            details.append(f"family annual income of **₹{merged['annual_income']:,.0f}**")
        if merged.get('government_school') == 'Yes':
            details.append("studied in **Government School (7.5% Quota)**")
        if merged.get('first_graduate') == 'Yes':
            details.append("**First Graduate** in family")

        summary_text = ", ".join(details) if details else "your academic details"
        
        reply = (
            f"{name_greet} Thank you for sharing your details. I've noted that you are {summary_text}. "
            "I have evaluated our verified scholarship database and tailored your top matching recommendations below!"
        )

        missing = []
        if not merged.get('annual_income'):
            missing.append("family income")
        if not merged.get('government_school'):
            missing.append("whether you attended a Govt school")
        if not merged.get('first_graduate'):
            missing.append("first graduate status")

        if missing:
            reply += f"\n\n💡 *Tip: You can also tell me your {missing[0]} in the next message to check for additional exclusive benefits!*"

        return reply


class CumulativeProfileManager:
    """Maintains cumulative session profile state with smart merging."""
    
    @staticmethod
    def get_initial_profile() -> Dict[str, Any]:
        return EMPTY_PROFILE.copy()

    @staticmethod
    def merge(current_profile: Dict[str, Any], new_attributes: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Merges new attributes into current profile without overwriting with null/empty values.
        Returns (updated_profile, list_of_changed_fields).
        """
        updated = current_profile.copy()
        changed = []
        
        for k, v in new_attributes.items():
            if k in EMPTY_PROFILE:
                if v is not None and str(v).strip() != "" and str(v).lower() != "null":
                    # Check if actually changed
                    if updated.get(k) != v:
                        updated[k] = v
                        changed.append(k)
                        
        return updated, changed

    @staticmethod
    def calculate_completion_percentage(profile: Dict[str, Any]) -> int:
        filled = sum(1 for k, v in profile.items() if v is not None and str(v).strip() != "")
        total = len(EMPTY_PROFILE)
        return int((filled / total) * 100)


class ScholarshipScoringEngine:
    """
    Computes precise, weighted match scores with strict filtering:
    - Gender strictness (never recommend female-exclusive to males, prioritize female-specific for females)
    - Student vs Parent disability strictness (avoid false positives on parent disability)
    - Community, Education Level, Income, Government School, First Graduate, Hosteller, and Merit scoring.
    """

    @classmethod
    def evaluate(cls, row: pd.Series, profile: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
        score = 0.0
        reasons = []
        breakdown = {}
        
        sch_name = str(row.get('Scholarship_Name', '')).strip()
        sch_desc = str(row.get('Eligibility_Description', '')).strip()
        sch_keywords = str(row.get('Keywords', '')).strip()
        sch_gender = str(row.get('Gender_Eligibility', '')).strip().lower()
        sch_disability = str(row.get('Disability_Eligibility', '')).strip().lower()
        sch_state = str(row.get('State', '')).strip().lower()
        sch_app_states = str(row.get('Applicable_States', '')).strip().lower()
        sch_comm = str(row.get('Community_Eligibility', '')).strip().lower()
        sch_edu = str(row.get('Education_Level', '')).strip().lower()
        sch_income = str(row.get('Annual_Income_Limit', '')).strip()
        sch_govt_sch = str(row.get('Government_School_Eligibility', '')).strip().lower()
        sch_first_grad = str(row.get('First_Graduate_Eligibility', '')).strip().lower()
        sch_hostel = str(row.get('Hosteller_Eligibility', '')).strip().lower()
        sch_marks = str(row.get('Minimum_Marks', '')).strip()
        sch_cgpa = str(row.get('Minimum_CGPA', '')).strip()
        
        combined_text = f"{sch_name} {sch_desc} {sch_keywords}".lower()

        # -------------------------------------------------------------
        # 1. GENDER ELIGIBILITY & STRICT FILTERING
        # -------------------------------------------------------------
        user_gender = (profile.get('gender') or '').strip().lower()
        is_sch_female_only = (
            'female' in sch_gender or 
            'girl' in sch_name.lower() or 
            'women' in sch_name.lower() or
            'pudhumai penn' in sch_name.lower() or 
            'nagammai' in sch_name.lower() or
            'rural girls' in sch_name.lower()
        )
        is_sch_male_only = ('male' in sch_gender and 'female' not in sch_gender)

        if user_gender:
            if user_gender == 'female':
                if is_sch_female_only:
                    pts = 15.0
                    score += pts
                    reasons.append("🎯 Female-Specific Scheme Priority")
                    breakdown['Gender'] = pts
                elif is_sch_male_only:
                    return -1.0, [], {}  # Ineligible
                else:
                    pts = 2.0
                    score += pts
                    reasons.append("✓ Open to all genders")
                    breakdown['Gender'] = pts
            elif user_gender == 'male':
                if is_sch_female_only:
                    return -1.0, [], {}  # STRICT DISQUALIFICATION
                elif is_sch_male_only:
                    pts = 15.0
                    score += pts
                    reasons.append("🎯 Male-Specific Scheme Match")
                    breakdown['Gender'] = pts
                else:
                    pts = 2.0
                    score += pts
                    reasons.append("✓ Open to all genders")
                    breakdown['Gender'] = pts

        # -------------------------------------------------------------
        # 2. DISABILITY FILTERING (STUDENT VS PARENT DISABILITY)
        # -------------------------------------------------------------
        user_dis = (profile.get('disability') or '').strip().lower()
        
        has_user_disability = user_dis and user_dis not in ['none', 'no', 'null', 'na', 'n/a', 'false', 'normal', 'not applicable']
        user_specifies_parent_disability = has_user_disability and any(k in user_dis for k in ['parent', 'father', 'mother', 'ward', 'son of', 'daughter of'])
        user_has_student_disability = has_user_disability and not user_specifies_parent_disability

        is_parent_disability_scheme = (
            'parent is differently abled' in sch_disability or
            'son/daughter of differently abled' in sch_name.lower() or
            'child of differently abled' in sch_hostel or
            'children of persons with disabilities' in sch_desc.lower()
        )
        is_student_disability_scheme = (
            'students with disabilities' in combined_text or
            ('differently abled' in combined_text and not is_parent_disability_scheme) or
            ('disabilit' in combined_text and not is_parent_disability_scheme)
        )

        if user_has_student_disability:
            if is_student_disability_scheme:
                pts = 25.0
                score += pts
                reasons.append(f"♿ Student Disability Scheme ({profile.get('disability')})")
                breakdown['Disability'] = pts
            elif is_parent_disability_scheme:
                # Heavy penalty: do NOT recommend parent-disability scheme to student with personal disability
                pts = -25.0
                score += pts
                breakdown['Disability'] = pts
        elif user_specifies_parent_disability:
            if is_parent_disability_scheme:
                pts = 30.0
                score += pts
                reasons.append("👨‍👩‍👧 Scheme for Wards/Children of Differently Abled Persons")
                breakdown['Disability'] = pts
            elif is_student_disability_scheme:
                pts = -15.0
                score += pts
                breakdown['Disability'] = pts
        else:
            # User does NOT have disability
            if is_student_disability_scheme or is_parent_disability_scheme:
                score -= 15.0  # Avoid recommending disability schemes to non-disabled students

        # -------------------------------------------------------------
        # 3. STATE & DOMICILE
        # -------------------------------------------------------------
        user_state = (profile.get('state') or '').strip().lower()
        if user_state:
            if 'tamil nadu' in user_state or user_state in ['tn', 'tamilnadu']:
                if 'tamil nadu' in sch_state or 'tamil nadu' in sch_app_states:
                    pts = 8.0
                    score += pts
                    reasons.append("📍 Tamil Nadu State Quota")
                    breakdown['State'] = pts
                elif 'india' in sch_state or 'india' in sch_app_states or 'all india' in sch_state:
                    pts = 5.0
                    score += pts
                    reasons.append("🇮🇳 All-India Scheme Eligibility")
                    breakdown['State'] = pts
            else:
                if user_state in sch_state or user_state in sch_app_states or 'india' in sch_state:
                    pts = 6.0
                    score += pts
                    reasons.append(f"📍 State Match ({profile.get('state')})")
                    breakdown['State'] = pts
                elif 'tamil nadu' in sch_state and 'india' not in sch_app_states:
                    score -= 15.0  # Out of state for TN-only scheme

        # -------------------------------------------------------------
        # 4. COMMUNITY
        # -------------------------------------------------------------
        user_comm = (profile.get('community') or '').strip().lower()
        if user_comm:
            # Map common variants
            c_key = user_comm
            if 'mbc' in user_comm:
                c_key = 'mbc'
            elif 'dnc' in user_comm:
                c_key = 'dnc'
            elif 'bc' in user_comm:
                c_key = 'bc'
            elif 'sc' in user_comm:
                c_key = 'sc'
            elif 'st' in user_comm:
                c_key = 'st'
            elif any(x in user_comm for x in ['oc', 'fc', 'general', 'open']):
                c_key = 'general'
            elif any(x in user_comm for x in ['minority', 'muslim', 'christian']):
                c_key = 'minority'

            # Evaluate against scheme
            if sch_comm and sch_comm not in ['', 'nan', 'any', 'not specified']:
                if c_key in sch_comm:
                    pts = 10.0
                    score += pts
                    reasons.append(f"👥 Community Match ({c_key.upper()})")
                    breakdown['Community'] = pts
                else:
                    # If scholarship is strictly for SC/ST and user is BC/MBC/OC, disqualify
                    if any(x in sch_comm for x in ['sc', 'st']) and c_key in ['bc', 'mbc', 'dnc', 'general']:
                        return -1.0, [], {}
                    elif 'minority' in sch_comm and c_key != 'minority':
                        return -1.0, [], {}
                    score -= 5.0
            else:
                # Check scheme title/keywords
                if f"– {c_key.upper()}" in sch_name or f" {c_key.upper()} " in sch_name:
                    pts = 10.0
                    score += pts
                    reasons.append(f"👥 Community Target ({c_key.upper()})")
                    breakdown['Community'] = pts
                elif c_key in ['bc', 'mbc', 'dnc'] and any(x in combined_text for x in ['obc', 'ebc', 'dnt']):
                    pts = 7.0
                    score += pts
                    reasons.append("👥 Central OBC/EBC category match")
                    breakdown['Community'] = pts
                elif c_key == 'sc' and 'sc' in combined_text and 'scholarship' not in combined_text:
                    pts = 8.0
                    score += pts
                    reasons.append("👥 SC category match")
                    breakdown['Community'] = pts
                elif c_key == 'st' and 'st' in combined_text:
                    pts = 8.0
                    score += pts
                    reasons.append("👥 ST category match")
                    breakdown['Community'] = pts
                else:
                    pts = 2.0
                    score += pts
                    reasons.append("👥 Open Community Scheme")
                    breakdown['Community'] = pts

        # -------------------------------------------------------------
        # 5. EDUCATION LEVEL
        # -------------------------------------------------------------
        user_edu = (profile.get('education_level') or '').strip().lower()
        if user_edu:
            is_ug = any(x in user_edu for x in ['ug', 'undergrad', 'degree', 'b.tech', 'be', 'b.sc', 'b.com', 'ba', 'bachelor', 'college'])
            is_pg = any(x in user_edu for x in ['pg', 'postgrad', 'master', 'm.tech', 'me', 'm.sc', 'm.com', 'mba', 'mca'])
            is_dip = any(x in user_edu for x in ['diploma', 'polytechnic'])
            is_iti = 'iti' in user_edu
            is_phd = any(x in user_edu for x in ['phd', 'doctorate', 'research'])
            is_sch = any(x in user_edu for x in ['school', '10th', '12th', 'plus two', 'pre-matric', 'class'])

            sch_combined_edu = (sch_edu + " " + combined_text).lower()
            
            # UG Check
            if is_ug:
                if any(x in sch_name.lower() for x in ['pre-matric', 'pre matric']):
                    score -= 15.0  # Pre-matric is school only, not for UG
                elif any(x in sch_combined_edu for x in ['ug', 'undergraduate', '3-year degree', 'college', 'higher education', 'technical degree', 'post-matric', 'top class']):
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 Undergraduate / College Level Match")
                    breakdown['Education'] = pts
                elif 'pg' in sch_combined_edu and 'ug' not in sch_combined_edu and 'higher education' not in sch_combined_edu:
                    score -= 10.0  # PG scheme not suited for UG
            # PG Check
            elif is_pg:
                if any(x in sch_combined_edu for x in ['pg', 'postgraduate', 'master', 'higher education']):
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 Postgraduate (PG) level match")
                    breakdown['Education'] = pts
                elif 'ug' in sch_combined_edu and 'pg' not in sch_combined_edu:
                    score -= 8.0
            # Diploma Check
            elif is_dip:
                if 'diploma' in sch_combined_edu or 'polytechnic' in sch_combined_edu:
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 Diploma level match")
                    breakdown['Education'] = pts
            # ITI Check
            elif is_iti:
                if 'iti' in sch_combined_edu:
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 ITI level match")
                    breakdown['Education'] = pts
            # School Check
            elif is_sch:
                if 'school' in sch_combined_edu or 'pre-matric' in sch_combined_edu or '6th' in sch_combined_edu:
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 School / Pre-Matric level match")
                    breakdown['Education'] = pts
            # PhD Check
            elif is_phd:
                if 'phd' in sch_combined_edu or 'research' in sch_combined_edu or 'fellowship' in sch_combined_edu:
                    pts = 8.0
                    score += pts
                    reasons.append("🎓 Research / PhD level match")
                    breakdown['Education'] = pts

        # -------------------------------------------------------------
        # 6. ANNUAL INCOME
        # -------------------------------------------------------------
        user_inc = profile.get('annual_income')
        if user_inc is not None and str(user_inc).strip() != "":
            try:
                user_inc_val = float(user_inc)
                parsed_limit = row.get('Parsed_Income_Limit')
                
                if parsed_limit is not None and parsed_limit > 0:
                    if user_inc_val <= parsed_limit:
                        pts = 6.0
                        score += pts
                        reasons.append(f"💰 Income verified (₹{user_inc_val:,.0f} ≤ ₹{parsed_limit:,.0f} limit)")
                        breakdown['Income'] = pts
                    else:
                        score -= 18.0  # Income exceeds required limit
                else:
                    if any(x in sch_income.lower() for x in ['no income', 'not specified', 'no economic', 'nan', '']):
                        pts = 2.0
                        score += pts
                        reasons.append("💰 No income limit barrier")
                        breakdown['Income'] = pts
            except:
                pass

        # -------------------------------------------------------------
        # 7. GOVERNMENT SCHOOL ELIGIBILITY
        # -------------------------------------------------------------
        user_govt_sch = (profile.get('government_school') or '').strip().lower()
        if user_govt_sch in ['yes', 'true']:
            if (
                'studied classes vi-xii in government schools' in sch_govt_sch or
                'government school' in combined_text or
                '7.5%' in combined_text or
                'pudhumai penn' in sch_name.lower()
            ):
                pts = 12.0
                score += pts
                reasons.append("🏫 Government school 7.5% / Special Quota Eligible")
                breakdown['GovtSchool'] = pts

        # -------------------------------------------------------------
        # 8. FIRST GRADUATE ELIGIBILITY
        # -------------------------------------------------------------
        user_first_grad = (profile.get('first_graduate') or '').strip().lower()
        if user_first_grad in ['yes', 'true']:
            if 'free education' in sch_name.lower() or 'first graduate' in combined_text or 'tuition fee waiver' in sch_name.lower():
                pts = 8.0
                score += pts
                reasons.append("🌟 First Graduate Fee Concession Priority")
                breakdown['FirstGraduate'] = pts

        # -------------------------------------------------------------
        # 9. HOSTELLER ELIGIBILITY
        # -------------------------------------------------------------
        user_hostel = (profile.get('hosteller') or '').strip().lower()
        if user_hostel in ['hosteller', 'hostel']:
            if 'hosteller' in sch_hostel or 'boarding/lodging' in sch_hostel or 'hostel' in combined_text:
                pts = 5.0
                score += pts
                reasons.append("🏠 Hosteller Maintenance / Boarding Support Included")
                breakdown['Hosteller'] = pts

        # -------------------------------------------------------------
        # 10. ACADEMIC MERIT (MARKS / CGPA)
        # -------------------------------------------------------------
        user_marks = profile.get('min_marks_or_cgpa')
        if user_marks is not None and str(user_marks).strip() != "":
            try:
                m_val = float(user_marks)
                if m_val > 10.0:  # Percentage
                    if m_val >= 60.0 and ('reliance' in combined_text or '60%' in sch_marks):
                        pts = 8.0
                        score += pts
                        reasons.append(f"📈 High Merit Qualified ({m_val:.1f}% ≥ 60% requirement)")
                        breakdown['Merit'] = pts
                    elif m_val >= 75.0:
                        pts = 3.0
                        score += pts
                        reasons.append(f"📈 Strong Academic Record ({m_val:.1f}%)")
                        breakdown['Merit'] = pts
                else:  # CGPA
                    if m_val >= 7.5 and ('7.5' in sch_cgpa or 'postgraduate' in combined_text):
                        pts = 8.0
                        score += pts
                        reasons.append(f"📈 CGPA Requirement Qualified ({m_val:.2f} ≥ 7.5)")
                        breakdown['Merit'] = pts
            except:
                pass

        return score, reasons, breakdown

    @classmethod
    def rank_scholarships(
        cls,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        top_n: int = 5,
        min_score: float = 5.0,
        provider_filter: Optional[str] = None,
        education_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Ranks all scholarships and returns top N matching recommendations."""
        recommendations = []
        
        for idx, row in df.iterrows():
            # Optional provider filter
            if provider_filter and provider_filter != "All":
                if row.get('Provider') != provider_filter:
                    continue

            # Optional education filter
            if education_filter and education_filter != "All":
                ed_str = str(row.get('Education_Level', '')).lower()
                if education_filter.lower() not in ed_str and 'nan' not in ed_str:
                    continue

            score, reasons, breakdown = cls.evaluate(row, profile)
            
            if score >= min_score:
                portal_url = str(row.get('Official_Source_URL', '')).strip()
                if not portal_url or portal_url == 'nan':
                    portal_url = str(row.get('Application_Portal', '')).strip()
                if not portal_url or portal_url == 'nan':
                    portal_url = "https://scholarships.gov.in"
                    
                recommendations.append({
                    "id": row.get('Scholarship_ID', f"SCH-{idx+1}"),
                    "name": row.get('Scholarship_Name', 'Scholarship Scheme'),
                    "provider": row.get('Provider', 'Government Authority'),
                    "amount": row.get('Scholarship_Amount', 'Varies / Full Tuition Waiver'),
                    "score": round(score, 1),
                    "reasons": reasons if reasons else ["General Eligibility Fit"],
                    "breakdown": breakdown,
                    "portal": portal_url,
                    "description": row.get('Eligibility_Description', ''),
                    "education_level": row.get('Education_Level', 'Higher Education'),
                    "state": row.get('State', 'All India'),
                    "documents": row.get('Documents_Required', 'Income Certificate, Community Certificate, Marksheets'),
                    "community": row.get('Community_Eligibility', 'Any')
                })

        # Sort descending by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]


class ScholarshipImprovementAdvisor:
    """Generates actionable, specific guidance to unlock higher scholarship scores and additional schemes."""
    
    @staticmethod
    def generate_guidance(profile: Dict[str, Any], current_recommendations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        tips = []
        
        # 1. Academic Merit Criteria
        marks = profile.get('min_marks_or_cgpa')
        if marks is None:
            tips.append({
                "category": "Academic Merit",
                "impact": "High Impact (+8 pts)",
                "icon": "📈",
                "title": "Add Your Academic Score (% or CGPA)",
                "description": "Scoring 60%+ in Class 12 or 7.5+ CGPA in College qualifies you for high-value merit schemes like Reliance Foundation (up to ₹2 Lakhs - ₹6 Lakhs)."
            })
        elif float(marks) < 60.0 and float(marks) > 10.0:
            deficit = 60.0 - float(marks)
            tips.append({
                "category": "Academic Merit",
                "impact": "High Impact (+8 pts)",
                "icon": "📈",
                "title": f"Target +{deficit:.1f}% to Reach 60% Cutoff",
                "description": "Reliance Foundation Undergraduate Scholarships require a minimum cutoff of 60% in Class 12. Reaching 60% unlocks up to ₹2,00,000 in financial support."
            })
        elif float(marks) < 7.5 and float(marks) <= 10.0:
            cgpa_diff = 7.5 - float(marks)
            tips.append({
                "category": "Academic Merit",
                "impact": "High Impact (+8 pts)",
                "icon": "📈",
                "title": f"Target +{cgpa_diff:.2f} CGPA to Reach 7.5 Threshold",
                "description": "Premier postgraduate scholarships and ICAR / Reliance PG fellowships require a 7.5 CGPA threshold."
            })

        # 2. Government School 7.5% Quota
        govt_sch = (profile.get('government_school') or '').strip().lower()
        if govt_sch != 'yes':
            tips.append({
                "category": "Government School Quota",
                "impact": "High Impact (+12 pts)",
                "icon": "🏫",
                "title": "Check 7.5% Government School Reservation",
                "description": "If you studied Classes VI to XII in Tamil Nadu Government schools, you qualify for 100% professional course fee reimbursement and Pudhumai Penn Thittam (₹1,000/month for female UG students)."
            })

        # 3. Income Ceiling & Documentation
        income = profile.get('annual_income')
        if income is None:
            tips.append({
                "category": "Income Verification",
                "impact": "Essential (+6 pts)",
                "icon": "💰",
                "title": "Declare Annual Family Income",
                "description": "Government Post-Matric schemes have an income ceiling of ₹2,50,000/year. Providing your income verifies full tuition and special fee exemptions."
            })
        elif float(income) > 250000.0:
            tips.append({
                "category": "Income Ceiling",
                "impact": "Eligibility Threshold",
                "icon": "💰",
                "title": "Verify Income Certificate Gross Ceiling",
                "description": "Your stated income exceeds ₹2,50,000. Review whether allowable deductions on your Tahsildar-issued Revenue Income Certificate qualify you for the ₹2.5L limit."
            })

        # 4. First Graduate Benefit
        first_grad = (profile.get('first_graduate') or '').strip().lower()
        if first_grad != 'yes':
            tips.append({
                "category": "First Graduate Benefit",
                "impact": "Fee Concession (+8 pts)",
                "icon": "🌟",
                "title": "Apply for First Graduate Certificate",
                "description": "If neither of your parents or elder siblings hold a degree, obtaining a First Graduate Certificate via TN e-Seva grants complete tuition fee waivers."
            })

        # 5. Hosteller Maintenance Support
        hostel = (profile.get('hosteller') or '').strip().lower()
        if hostel != 'hosteller':
            tips.append({
                "category": "Accommodation Grant",
                "impact": "Maintenance (+5 pts)",
                "icon": "🏠",
                "title": "Hosteller Maintenance Allowance",
                "description": "Students staying in recognized college hostels or government welfare hostels receive additional boarding & lodging stipends."
            })

        # 6. Community & DBT Verification
        comm = profile.get('community')
        if comm in ['BC', 'MBC', 'DNC', 'SC', 'ST']:
            tips.append({
                "category": "Direct Benefit Transfer (DBT)",
                "impact": "Compliance",
                "icon": "🏛️",
                "title": f"Aadhaar-Bank Seeding for {comm} Welfare",
                "description": f"Ensure your Aadhaar is linked to your bank account with active NPCI mapping for direct scholarship disbursement via the Tamil Nadu UMIS portal."
            })

        return tips
