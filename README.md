# Vorak: Vulnerability Oriented Red-teaming for AI Knowledge

[![PyPI version](https://badge.fury.io/py/vorak.svg)](https://pypi.org/project/vorak/1.0.0/)
[![License](https://img.shields.io/github/license/ruchirk22/vorak.svg)](https://github.com/ruchirk22/vorak/blob/main/LICENSE.txt)

Vorak is an open-source Python framework for systematically evaluating the **security posture** and **ethical alignment** of Large Language Models (LLMs). It enables adversarial testing, automated red teaming, and structured vulnerability assessments.

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
pip install vorak
```

For Development setup (cloning and local installation):

```bash
git clone [https://github.com/ruchirk22/vorak.git](https://github.com/ruchirk22/vorak.git)
cd vorak
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
vorak evaluate --model "openrouter/google/gemma-2-9b-it:free" --prompt-id "JBR_001"
```

Batch Evaluation:

```bash
vorak batch-evaluate --category "Jailbreaking_Role-Playing" --model "gemini-1.5-flash-latest" --output-json results.json
```

### Web Interface

Start the streamlit-based UI:

```bash
streamlit run vorak/web_interface/Home.py
```

## Project Structure

```bash
vorak/               # Core framework
tests/               # Unit tests
.github/workflows/   # CI/CD configurations
pyproject.toml       # Build configuration
requirements.txt     # Python dependencies
CONTRIBUTING.md      # Contribution guidelines
LICENSE.txt          # License
README.md            # Project documentation
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

vorak is licensed under the Apache 2.0 License.

## Citation

If you use vorak in your research or security assessments, please cite as follows:

```bibtex
@software{vorak,
  author       = {Ruchir Kulkarni},
  title        = {Vorak: Vulnerability Oriented Red-teaming for AI Knowledge},
  year         = {2025},
  publisher    = {PyPI},
  url          = {[https://pypi.org/project/vorak/](https://pypi.org/project/vorak/)}
}
```
