# setup.py

from setuptools import setup, find_packages

setup(
    name="aegis-framework",
    version="1.0.0",
    author="Ruchir Kulkarni", # You can change this
    author_email="mailtoruchirk@gmail.com", # And this
    description="Aegis: An open-source LLM Red Teaming and Evaluation Framework.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ruchirk22/aegis", # Add your GitHub repo URL here
    packages=find_packages(),
    install_requires=[
        "typer[all]",
        "rich",
        "python-dotenv",
        "openai",
        "google-generativeai",
        "pandas",
        "streamlit",
        "plotly",
        "fpdf2",
        "kaleido"
    ],
    entry_points={
        'console_scripts': [
            'aegis = aegis.cli:app',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Or another license of your choice
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
