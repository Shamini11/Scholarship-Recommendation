# Entry point forwarding to webapp.py
import runpy

if __name__ == "__main__":
    runpy.run_path("webapp.py", run_name="__main__")
