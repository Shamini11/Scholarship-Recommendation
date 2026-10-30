"""
app.py - Dual-Mode Entry Point:
1. When launched via `streamlit run app.py`: Launches the Streamlit Web Application.
2. When launched via `python app.py`: Launches the interactive Terminal Assistant.
"""

import sys
import runpy

def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        pass
    
    for arg in sys.argv:
        if 'streamlit' in arg.lower():
            return True
    return False

if is_running_in_streamlit():
    # Execute full web application
    runpy.run_path("webapp.py", run_name="__main__")
else:
    # Interactive CLI Mode
    import json
    from engine import (
        ScholarshipDataLoader,
        GeminiNLPExtractor,
        CumulativeProfileManager,
        ScholarshipScoringEngine,
        DEFAULT_GEMINI_API_KEY,
        DEFAULT_MODEL_NAME
    )

    print("=" * 70)
    print("🎓 AI SCHOLARSHIP RECOMMENDATION & ELIGIBILITY SYSTEM (CLI & WEB)")
    print("=" * 70)
    print("Loading scholarship dataset...")
    loader = ScholarshipDataLoader()
    print(f"Loaded {len(loader.df)} scholarships successfully.\n")

    extractor = GeminiNLPExtractor(DEFAULT_GEMINI_API_KEY, DEFAULT_MODEL_NAME)
    profile = CumulativeProfileManager.get_initial_profile()

    print("Tell me about yourself in natural language.")
    print("Example: 'I am a female BC student studying UG in Tamil Nadu with an annual family income of 2.2 lakhs.'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'stop']:
                print("\nGood luck with your scholarship applications!")
                break

            print("\n[AI]: Processing your message with Gemini NLP & Cumulative Memory...")
            extracted_data, status = extractor.extract_with_gemini(user_input, profile)
            profile, changed = CumulativeProfileManager.merge(profile, extracted_data)

            print(f"[Engine Status]: {status}")
            print(f"[Cumulative Profile Snapshot]:")
            for k, v in profile.items():
                if v is not None:
                    print(f"  • {k}: {v}")

            recommendations = ScholarshipScoringEngine.rank_scholarships(loader.df, profile, top_n=5)
            print(f"\n--- Top {len(recommendations)} Recommended Scholarships ---")
            if not recommendations:
                print("No matching scholarships found. Try providing more details.")
            else:
                for i, sch in enumerate(recommendations, 1):
                    print(f"\n{i}. {sch['name']} ({sch['provider']})")
                    print(f"   - Match Score: {sch['score']} points")
                    print(f"   - Amount / Benefit: {sch['amount']}")
                    print(f"   - Matched Reasons: {', '.join(sch['reasons'])}")
                    print(f"   - Official Portal: {sch['portal']}")

            print("\n" + "-" * 70)
            print(">>> [Tip]: Add more info in the next sentence or type 'exit' to quit.")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"[Error]: {e}")
