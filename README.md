# Aegis: An Open-Source LLM Red Teaming & Evaluation Framework

[![PyPI version](https://badge.fury.io/py/aegisred.svg)](https://pypi.org/project/aegisred/)
[![Python versions](https://img.shields.io/pypi/pyversions/aegisred.svg)](https://pypi.org/project/aegisred/)
[![License](https://img.shields.io/github/license/ruchirk22/aegis.svg)](LICENSE.txt)
[![Downloads](https://pepy.tech/badge/aegisred)](https://pepy.tech/project/aegisred)

Aegis is an open-source Python framework for systematically evaluating the **security posture** and **ethical alignment** of Large Language Models (LLMs). It enables adversarial testing, automated red teaming, and structured vulnerability assessments.  

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface-cli)
  - [Web Interface](#web-interface)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Community & Support](#community--support)

---

## Features

- **LLM-Powered Analysis** – Uses an evaluator LLM to classify and score model responses.  
- **Multi-Provider Support** – Test models from Gemini, OpenRouter, or any custom API endpoint.  
- **Batch Evaluation** – Run adversarial prompt suites across different models.  
- **Comprehensive Reporting** – Export results as PDF, JSON, or CSV.  
- **Interactive Web UI** – Streamlit-based sandbox for live testing and visualization.  

---

## Installation

Install directly from PyPI:

```bash
pip install aegisred
```

For Development setup (cloning and local installation):

```bash
git clone https://github.com/ruchirk22/aegis.git
cd aegis
pip install -r requirements.txt
pip install -e .
```

## Configuration

Set provider API key for LLM Evaluation.
Create a .env in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### Command Line Interface (CLI)

Single Prompt Evaluation:

```bash
aegisred evaluate --model "openrouter/google/gemma-2-9b-it:free" --prompt-id "JBR_001"
```

Batch Evaluation:

```bash
aegisred batch-evaluate --category "Jailbreaking_Role-Playing" --model "gemini-1.5-flash-latest" --output-json results.json
```

### Web Interface

Start the streamlit-based UI:

```bash
streamlit run aegis/web_interface/Aegis.py
```

## Project Structure

```bash
aegis/               # Core framework
tests/               # Unit tests
.github/workflows/   # CI/CD configurations
pyproject.toml       # Build configuration
requirements.txt     # Python dependencies
CONTRIBUTING.md      # Contribution guidelines
LICENSE.txt          # License
README.md            # Project documentation
```

## Contributing

We welcome contributions from the community.
Please see CONTRIBUTING.md for setup instructions, coding standards, and best practices.

- Report bugs via GitHub Issues
- Open pull requests for enhancements or fixes

## License

Aegis is licensed under the Apache 2.0 License.

## Citation

If you use Aegis in your research or security assessments, please cite as follows:

```bibtex
@software{aegisred,
  author       = {Ruchir Kulkarni},
  title        = {Aegis: LLM Red Teaming & Evaluation Framework},
  year         = {2025},
  publisher    = {PyPI},
  url          = {https://pypi.org/project/aegisred/}
}
```

## Acknowledgements

- Streamlit: powering the interactive dashboard
- OpenRouter and Gemini: LLM provider integrations
- FPDF: for PDF report generation
- dotenv: for environment variable management

## Community & Support

- Github Repository
- Issue Tracker
- PyPi Package

For questions, discussions, or collaboration, please open an issue or reach out via GitHub.
