# 🛡️ Aegis: An Open-Source LLM Red Teaming & Evaluation Framework

Aegis is a comprehensive, Python-based toolkit for systematically evaluating the security posture and ethical alignment of Large Language Models (LLMs) through adversarial testing and vulnerability assessment.

## Key Features

- **🤖 LLM-Powered Analysis**: Uses a powerful LLM to intelligently classify model responses.
- **🔌 Multi-Provider Support**: Test models from Gemini, OpenRouter, and any custom API endpoint.
- **🚀 Batch Evaluation**: Run entire categories of adversarial prompts against a model at once.
- **📊 Comprehensive Reporting**: Generate detailed PDF, JSON, and CSV reports for analysis.
- **💻 Interactive Web UI**: A Streamlit-based sandbox for live testing and a dashboard for visualizing results.

## 🚀 Quick Start

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ruchirk22/aegis.git
cd aegis-framework
pip install -r requirements.txt
```

> **Note:** Once published to PyPI, you will be able to install with:

 ```bash
 pip install aegis-framework
 ```

### Set Up API Keys

Set your API keys as environment variables before running evaluations:

```bash
export GEMINI_API_KEY=your_gemini_api_key
export OPENROUTER_API_KEY=your_openrouter_api_key
```

Or create a `.env` file in the project root:

``` bash
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Run an Evaluation from the CLI

1. **Run a single prompt evaluation:**

    ```bash
    python -m aegis evaluate --model "openrouter/google/gemma-2-9b-it:free" --prompt-id "JBR_001"
    ```

2. **Run a batch evaluation by category:**

    ```bash
    python -m aegis batch-evaluate --category "Jailbreaking_Role-Playing" --model "gemini-1.5-flash-latest" --output-json jailbreak_report.json
    ```

### Launch the Web Interface

Start the Streamlit web UI for interactive testing and dashboards:

```bash
streamlit run aegis/web_interface/Aegis.py
```

To view the Security Dashboard:

```bash
streamlit run aegis/web_interface/pages/1_Security_Dashboard.py
```

## 📚 Documentation

Full usage guides and API references are coming soon. For now, see the example commands above and explore the [aegis/core](aegis/core/) and [aegis/web_interface](aegis/web_interface/) directories for implementation details.

## 🤝 Contributing

We welcome contributions! Please open issues or submit pull requests on [GitHub](https://github.com/ruchirk22/aegis). For setup instructions and contribution guidelines, see `CONTRIBUTING.md` (to be added).

## 📜 License

Aegis is licensed under the [Apache 2.0 License](LICENSE.txt)
