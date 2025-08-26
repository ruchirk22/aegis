# Contributing to Vorak

First off, thank you for considering contributing to Vorak! It's people like you that make open-source software such a powerful force for good in the world. Every contribution, from reporting a bug to a new feature, helps make the AI ecosystem safer. This document provides clear guidelines to help you get started.

## How Can I Contribute?

There are many ways to contribute to Vorak, and all are valuable:

- **Reporting Bugs**: Open an issue on our GitHub Issues page. Please include:
  - A clear title and description
  - Steps to reproduce
  - Expected vs. actual behavior
- **Suggesting Enhancements**: Have an idea for a new feature or an improvement to an existing one? Open an issue to start a discussion.
- **Contributing New Prompts (Core of Vorak)**: Vorak thrives on its adversarial prompt library. We have made it easy to add your own prompts.
- **Writing Documentation**: Clear documentation is crucial. If you spot an area for improvement, feel free to update it or suggest changes.
- **Submitting Pull Requests (Code Contributions)**: If you are a developer, you can directly contribute to the codebase.

## Submitting New Prompts

We have created a built-in CLI command to make contributing new prompts straightforward.

### Step 1: Create a JSON File

Create a JSON file containing one or more prompts. Use this exact structure:

```json
[
  {
    "id": "YOUR_UNIQUE_ID",
    "category": "One_Of_The_Existing_Categories",
    "subcategory": "A_Descriptive_Subcategory",
    "severity": "LOW/MEDIUM/HIGH/CRITICAL",
    "prompt_text": "The text of your adversarial prompt.",
    "expected_behavior": "REJECT",
    "tags": ["new-tag", "another-tag"]
  }
]
```

Notes:

- **id**: Must be unique and not exist in the current library.
- **category**: Must match an existing category (for example: Jailbreaking_Role-Playing, Code_Generation_Exploits).

### Step 2: Validate Your File

Use the built-in CLI command to validate:

```bash
vorak prompt contribute --file /path/to/your/new_prompts.json
```

### Step 3: Create a GitHub Issue

If validation passes, the command will output a Markdown block. Copy this block and paste it into a new issue on GitHub. Our team will review your submission, and once approved, it will be merged into the library in a future release.

## Development Setup

If you wish to contribute code:

1. Fork the repository on GitHub.

2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/vorak.git
   cd vorak
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies (editable mode):

   ```bash
   pip install -e ".[local]"
   ```

5. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

6. Make your changes and run tests:

   ```bash
   pytest
   ```

## Pull Request Process

1. Ensure build and install dependencies are removed before finalizing.
2. Update README.md if your changes affect the interface, environment variables, ports, or usage.
3. Create a new issue to discuss the bug or feature you are working on.
4. Push your branch and open a pull request from your fork to the main repository.
5. Link the pull request to the corresponding issue.
6. Use clear and descriptive commit messages.

We will review your pull request as soon as possible.

## Code of Conduct

### Our Pledge

We pledge to make participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate corrective action in response to instances of unacceptable behavior.

### Scope

This Code of Conduct applies within project spaces and in public spaces when an individual is representing the project or its community.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by opening an issue or contacting the maintainers directly. All complaints will be reviewed and investigated, and will result in a response that is appropriate to the circumstances. Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent consequences as determined by the project team.

## Thank You

Thank you again for your interest in making Vorak the best AI red-teaming framework in the world. Your contributions, big or small, make a real impact.
