# vorak/__main__.py

# This file is the sole entry point for the command line.
# It imports the Typer app from cli.py and runs it.

from vorak.cli import app

if __name__ == "__main__":
    app()
