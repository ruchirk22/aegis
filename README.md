# Vorak: Vulnerability Oriented Red-teaming for AI Knowledge

[![PyPI version](https://badge.fury.io/py/vorak.svg)](https://pypi.org/project/vorak/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/vorak.svg)](https://pypi.org/project/vorak/)

**Vorak** is an advanced, enterprise-grade framework for systematically evaluating the **security, safety, and compliance** of Large Language Models (LLMs). It moves beyond static testing by using AI-driven techniques to discover novel vulnerabilities, providing a comprehensive solution for researchers, developers, and enterprises to secure their generative AI systems.

---

## Table of Contents

- [Why Vorak?](#why-vorak)
- [Core Features](#core-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Web Interface](#web-interface)
- [Contributing](#contributing)
- [License](#license)

---

## Why Vorak?

While many tools can test for known vulnerabilities, Vorak is designed to discover the unknown. Its intelligent, multi-layered approach provides a deeper and more realistic assessment of AI security posture.

- **Automated Attack Escalation**: Automatically discovers new attack vectors when a model resists initial attempts.
- **Enterprise-Ready Compliance**: Translates security findings into actionable compliance risks for frameworks like NIST, EU AI Act, and MITRE ATLAS.
- **Progress Tracking**: Provides concrete data to measure and report on security improvements over time.
- **Unified Experience**: Offers a seamless workflow for both developers (CLI) and security teams (UI).

---

## Core Features

- **Adaptive, AI-Powered Attack Escalation**: If a model is compliant, Vorak's `adaptive` mode uses an LLM to automatically generate a stronger, more sophisticated prompt to bypass defenses.

- **Multi-Turn Scenario Testing**: Simulates complex conversational attacks where vulnerabilities emerge over several turns, powered by an AI "scenario strategist".

- **Integrated Governance Layer**: Automatically maps detected vulnerabilities to major compliance frameworks, including **NIST AI RMF**, **EU AI Act**, **ISO/IEC 23894**, and **MITRE ATLAS**.

- **Comparative Reporting**: Compare two test sessions to track security posture over time, identify regressions, and generate executive-ready PDF reports showing the delta.

- **Security Sandbox**: Safely analyzes generated code for dangerous patterns (e.g., file system access, network calls) using **static analysis** without ever executing the code.

- **Advanced Prompt Generation**: Uses Gemini to augment and create novel adversarial prompts with sophisticated strategies like `adversarial_phrasing`.

- **YAML-Based Playbooks**: Orchestrate complex, multi-step testing sequences for repeatable and shareable evaluation workflows.

- **Community Contribution CLI**: A dedicated command (`vorak prompt contribute`) to validate and format new prompts, making it easy for the community to expand the prompt library.

---

## Installation

Install the latest release from PyPI:

```bash
pip install vorak
```

For users who want to test local models (e.g., from Hugging Face), install the optional `local` dependencies:

```bash
pip install "vorak[local]"
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/ruchirk22/vorak.git
cd vorak
pip install -e .
```

---

## Configuration

Vorak uses a `.env` file in your project's root directory to manage API keys.

```bash
# .env file
GEMINI_API_KEY="your_gemini_api_key"
OPENAI_API_KEY="your_openai_api_key"
ANTHROPIC_API_KEY="your_anthropic_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
TAVILY_API_KEY="your_tavily_api_key" # Required for agent testing
```

---

## Quick Start

Get help on any command:

```bash
vorak --help
vorak evaluate --help
vorak prompt --help
```

Run a standard evaluation:

```bash
vorak evaluate -p "JBR_001" -m "gemini/gemini-1.5-flash-latest"
```

Run an adaptive evaluation and save a report:

```bash
vorak evaluate -p "CGE_001" -m "gemini/gemini-1.5-flash-latest" --mode adaptive --output-pdf adaptive_report.pdf
```

---

## Usage

### Command-Line Interface (CLI)

#### **Running Evaluations**

```bash
# Standard evaluation with governance mapping
vorak evaluate -p "DPI_001" -m "openai/gpt-4o-mini" --mode governance

# Multi-turn scenario test
vorak evaluate -p "JBR_002" -m "anthropic/claude-3-5-sonnet-20240620" --mode scenario --turns 4
```

#### **Comparing Sessions**

```bash
# First, run two batch evaluations to get session IDs
vorak batch-evaluate -c "Code_Generation_Exploits" -m "gemini/gemini-1.5-flash-latest"
vorak batch-evaluate -c "Code_Generation_Exploits" -m "openrouter/google/gemma-2-9b-it:free"

# Then, compare the results
vorak compare-sessions "batch_..." "batch_..." --output-pdf comparison.pdf
```

#### **Managing Prompts**

```bash
# Generate 5 new prompts using advanced AI-driven phrasing
vorak prompt generate -c "Misinformation_Deception" -n 5 --strategy adversarial_phrasing -o new_prompts.json

# Validate and prepare prompts for a GitHub contribution
vorak prompt contribute -f new_prompts.json

# Clean up the main library file
vorak prompt cleanup
```

#### **Executing Playbooks**

```bash
# Run a pre-defined sequence of tests from a YAML file
vorak run --playbook my_security_plan.yaml
```

### Web Interface

Launch the interactive Streamlit UI for a full GUI experience, including a security dashboard and session comparison tools.

```bash
streamlit run vorak/web_interface/Home.py
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to submit prompts, report bugs, or add new features.

---

## License

Vorak is licensed under the [Apache 2.0 License](LICENSE).
