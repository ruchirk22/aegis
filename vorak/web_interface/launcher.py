# vorak/web_interface/launcher.py

import os
import sys
from streamlit.web import cli as stcli

def main():
    """
    This is the entry point for the 'vorak-ui' command.
    It locates and runs the main Streamlit app file.
    """
    # Get the directory where this launcher script is located
    current_dir = os.path.dirname(__file__)
    # Construct the full path to the Home.py file
    app_path = os.path.join(current_dir, "Home.py")
    
    # Check if the file exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find the main application file at {app_path}")
        sys.exit(1)
        
    # Use the Streamlit CLI to run the app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
