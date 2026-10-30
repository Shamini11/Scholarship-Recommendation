import sys
sys.stdout.reconfigure(encoding='utf-8')
import unittest
from engine import (
    ScholarshipDataLoader,
    RuleBasedNLPExtractor,
    CumulativeProfileManager,
    ScholarshipScoringEngine,
    EMPTY_PROFILE
)

class TestScholarshipSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = ScholarshipDataLoader()
        cls.df = cls.loader.df

    def test_01_dataset_loaded(self):
        self.assertEqual(len(self.df), 72, "Dataset should have 72 rows")
        expected_cols = [
            'Scholarship_Name', 'Provider', 'Scholarship_Amount', 'State',
            'Applicable_States', 'Community_Eligibility', 'Education_Level',
            'Annual_Income_Limit', 'Gender_Eligibility', 'Disability_Eligibility',
            'Government_School_Eligibility', 'First_Graduate_Eligibility',
            'Hosteller_Eligibility', 'Official_Source_URL'
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_02_nlp_extraction_keys(self):
        text = "I am a female BC student in Tamil Nadu with a walking disability and 85% marks."
        extracted = RuleBasedNLPExtractor.extract(text)
        self.assertEqual(extracted.get('gender'), 'Female')
        self.assertEqual(extracted.get('community'), 'BC')
        self.assertEqual(extracted.get('state'), 'Tamil Nadu')
        self.assertIn('Walking Disability', extracted.get('disability', ''))
        self.assertEqual(extracted.get('min_marks_or_cgpa'), 85.0)

    def test_03_cumulative_memory_merge(self):
        p0 = CumulativeProfileManager.get_initial_profile()
        # Turn 1
        t1 = "I am a female BC student studying UG in Tamil Nadu."
        ext1 = RuleBasedNLPExtractor.extract(t1)
        p1, ch1 = CumulativeProfileManager.merge(p0, ext1)
        
        self.assertEqual(p1['gender'], 'Female')
        self.assertEqual(p1['community'], 'BC')
        self.assertEqual(p1['state'], 'Tamil Nadu')
        self.assertEqual(p1['education_level'], 'UG')

        # Turn 2: add income and first graduate
        t2 = "My annual family income is 2 lakhs and I am the first graduate."
        ext2 = RuleBasedNLPExtractor.extract(t2)
        p2, ch2 = CumulativeProfileManager.merge(p1, ext2)

        # Check turn 1 retained
        self.assertEqual(p2['gender'], 'Female')
        self.assertEqual(p2['community'], 'BC')
        self.assertEqual(p2['state'], 'Tamil Nadu')
        self.assertEqual(p2['education_level'], 'UG')
        # Check turn 2 added
        self.assertEqual(p2['annual_income'], 200000.0)
        self.assertEqual(p2['first_graduate'], 'Yes')

    def test_04_strict_disability_filtering(self):
        # Case A: Student has walking disability
        student_prof = {
            'gender': 'Female',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'disability': 'Walking Disability',
            'annual_income': 200000.0
        }
        recs_student = ScholarshipScoringEngine.rank_scholarships(self.df, student_prof, top_n=10)
        rec_names = [r['name'] for r in recs_student]
        
        # Student disability schemes should appear
        has_student_scheme = any('Students with Disabilities' in name for name in rec_names)
        self.assertTrue(has_student_scheme, "Expected student disability scheme in recommendations")
        
        # Son/Daughter of Differently Abled Persons should NOT be in top 3
        top_3_names = [r['name'] for r in recs_student[:3]]
        self.assertNotIn("Scholarship to Son/Daughter of Differently Abled Persons – UG", top_3_names)

        # Case B: Parent has disability
        parent_prof = {
            'gender': 'Male',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'disability': 'Parent is differently abled',
            'annual_income': 150000.0
        }
        recs_parent = ScholarshipScoringEngine.rank_scholarships(self.df, parent_prof, top_n=5)
        top_parent_names = [r['name'] for r in recs_parent]
        self.assertIn("Scholarship to Son/Daughter of Differently Abled Persons – UG", top_parent_names)

    def test_05_strict_gender_filtering(self):
        # Male student must NEVER be recommended female-only schemes
        male_prof = {
            'gender': 'Male',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'government_school': 'Yes'
        }
        recs_male = ScholarshipScoringEngine.rank_scholarships(self.df, male_prof, top_n=20)
        male_rec_names = [r['name'] for r in recs_male]
        self.assertNotIn("Pudhumai Penn Thittam", male_rec_names, "Pudhumai Penn Thittam is female-only")
        self.assertNotIn("EVR Nagammai Scholarship", male_rec_names, "EVR Nagammai is female-only")
        self.assertNotIn("Rural Girls Incentive Scheme – MBC/DNC", male_rec_names)

        # Female student should have Pudhumai Penn Thittam recommended when govt school is Yes
        female_prof = {
            'gender': 'Female',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'government_school': 'Yes'
        }
        recs_female = ScholarshipScoringEngine.rank_scholarships(self.df, female_prof, top_n=5)
        female_rec_names = [r['name'] for r in recs_female]
        self.assertIn("Pudhumai Penn Thittam", female_rec_names)

    def test_06_recommendation_details_and_links(self):
        prof = {
            'gender': 'Female',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'annual_income': 200000.0
        }
        recs = ScholarshipScoringEngine.rank_scholarships(self.df, prof, top_n=5)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn('name', r)
            self.assertIn('provider', r)
            self.assertIn('amount', r)
            self.assertIn('score', r)
            self.assertIn('reasons', r)
            self.assertIn('portal', r)
            self.assertTrue(r['portal'].startswith('http'), f"Invalid portal link: {r['portal']}")
            self.assertGreater(len(r['reasons']), 0, "Reasons list should not be empty")

    def test_07_auth_registration_and_login(self):
        from auth import register_user, authenticate_user
        # Register test account
        email = "test.scholar@example.com"
        ok, msg = register_user(
            name="Priya Sharma",
            phone="9876543210",
            email=email,
            password="StrongPassword@123",
            confirm_password="StrongPassword@123"
        )
        self.assertTrue(ok or "already exists" in msg, f"Registration failed: {msg}")

        # Authenticate
        auth_ok, user_profile, auth_msg = authenticate_user(email, "StrongPassword@123")
        self.assertTrue(auth_ok, f"Authentication failed: {auth_msg}")
        self.assertEqual(user_profile["email"], email)
        self.assertEqual(user_profile["name"], "Priya Sharma")

    def test_08_password_mismatch_and_validation(self):
        from auth import register_user
        # Mismatched passwords
        ok, msg = register_user(
            name="Priya Sharma",
            phone="9876543210",
            email="mismatch@example.com",
            password="Password1",
            confirm_password="Password2"
        )
        self.assertFalse(ok)
        self.assertIn("do not match", msg.lower())

    def test_09_voice_transcript_nlp_extraction(self):
        # Test that text coming from a voice dictation is cleanly extracted
        voice_transcript = "I am studying UG in Tamil Nadu and my parent is differently abled. We are day scholars."
        extracted = RuleBasedNLPExtractor.extract(voice_transcript)
        self.assertEqual(extracted.get('education_level'), 'UG')
        self.assertEqual(extracted.get('state'), 'Tamil Nadu')
        self.assertEqual(extracted.get('disability'), 'Parent is differently abled')
        self.assertEqual(extracted.get('hosteller'), 'Day Scholar')

    def test_10_what_if_and_improvement_advisor(self):
        from engine import ScholarshipImprovementAdvisor
        # Test base profile
        base_prof = {
            'gender': 'Female',
            'community': 'BC',
            'state': 'Tamil Nadu',
            'education_level': 'UG',
            'annual_income': 280000.0,
            'min_marks_or_cgpa': 55.0,
            'government_school': 'No',
            'first_graduate': 'No'
        }
        recs = ScholarshipScoringEngine.rank_scholarships(self.df, base_prof, top_n=5)
        
        # Test guidance generation
        tips = ScholarshipImprovementAdvisor.generate_guidance(base_prof, recs)
        self.assertGreater(len(tips), 0, "Expected actionable tips")
        tip_categories = [t['category'] for t in tips]
        self.assertIn('Academic Merit', tip_categories)
        self.assertIn('Government School Quota', tip_categories)

        # Test What-If simulation (raising marks to 85% and income under 2.5L)
        simulated = base_prof.copy()
        simulated['min_marks_or_cgpa'] = 85.0
        simulated['annual_income'] = 200000.0
        sim_recs = ScholarshipScoringEngine.rank_scholarships(self.df, simulated, top_n=5)
        self.assertGreater(sim_recs[0]['score'], recs[0]['score'], "What-If top score should be higher")

if __name__ == '__main__':
    unittest.main(verbosity=2)
