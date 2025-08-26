# Vorak: Vulnerability Oriented Red-teaming for AI Knowledge

[![PyPI version](https://badge.fury.io/py/vorak.svg)](https://pypi.org/project/vorak/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/vorak.svg)](https://pypi.org/project/vorak/)

Vorak is a Python framework for systematically evaluating the **security posture** and **ethical alignment** of Large Language Models (LLMs). It enables adversarial testing, automated red-teaming, and structured vulnerability assessments to help researchers, developers, and enterprises identify and mitigate risks in generative AI systems.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Web Interface](#web-interface)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **LLM-Powered Evaluation** – Leverages evaluator LLMs to classify, score, and explain model behavior.
- **Multi-Provider Support** – Compatible with Gemini, OpenRouter, Anthropic, OpenAI, and custom APIs.
- **Adversarial Prompt Libraries** – Run curated or custom adversarial test suites at scale.
- **Batch & Automated Testing** – Evaluate multiple models, prompts, or categories in a single run.
- **Risk & Alignment Reporting** – Export structured outputs as **PDF, JSON, or CSV** for audits and dashboards.
- **Interactive Web UI** – Streamlit-powered sandbox for interactive testing and visualization.

---

## Installation

Install the latest release from PyPI:

```bash
pip install vorak
```

If you want to test local models as well, download optional dependencies:

```bash
pip install "vorak[local]"
# This will install heavy dependencies
```

For development setup (cloning and local installation):

```bash
git clone https://github.com/ruchirk22/vorak.git
cd vorak
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

For help regarding all commands:

```bash
vorak --help
vorak [COMMAND] --help #for specific command-related help
```

Run a single prompt evaluation:

```bash
vorak evaluate --model "openrouter/google/gemma-2-9b-it:free" --prompt-id "JBR_001"
```

Run a batch evaluation across categories:

```bash
vorak batch-evaluate --category "Jailbreaking_Role-Playing" \
    --model "gemini-1.5-flash-latest" \
    --output-json results.json
```

If you are using local models, make sure to install the required dependencies.

---

## Configuration

Vorak uses environment variables for model provider authentication. Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

---

## Usage

### Command-Line Interface

View available commands:

```bash
vorak --help
vorak [COMMAND] --help #for specific command-related help
```

### Web Interface

Launch the interactive Streamlit UI:

```bash
streamlit run vorak/web_interface/Home.py
```

---

## Project Structure

```bash
vorak/               # Core framework
  ├── cli.py         # CLI entrypoint
  ├── agents/        # Attack agents
  ├── core/          # Evaluators, connectors, analyzers
  ├── prompts/       # Prompt libraries
  └── web_interface/ # Streamlit-based UI

.tests/              # Unit tests
.github/workflows/   # CI/CD configs
pyproject.toml       # Build configuration
requirements.txt     # Dependencies
CONTRIBUTING.md      # Contribution guidelines
LICENSE              # License
README.md            # Documentation
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Vorak is licensed under the [Apache 2.0 License](LICENSE).

---

## Citation

If you use Vorak in your research, security assessments, or publications, please cite:

```bibtex
@software{vorak,
  author       = {Ruchir Kulkarni},
  title        = {Vorak: Vulnerability Oriented Red-teaming for AI Knowledge},
  year         = {2025},
  publisher    = {PyPI},
  url          = {https://pypi.org/project/vorak/}
}
```
