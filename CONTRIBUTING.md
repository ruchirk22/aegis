# Contributing to Aegis

We're thrilled you're interested in contributing to Aegis! Every contribution—from a bug report to a new feature—helps make the AI ecosystem safer. This guide will help you get started.

## How to Contribute

You can contribute in several ways:

* **Reporting Bugs:** If you find a bug, please open an issue on our GitHub repository. Include a clear title, a description of the bug, and steps to reproduce it.
* **Suggesting Enhancements:** Have an idea for a new feature or an improvement? Open an issue to start a discussion.
* **Submitting Pull Requests:** If you want to contribute code, please follow the process below.

## 💻 Setting Up Your Development Environment

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/aegis.git
   cd aegis
   ```

3. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

4. **Install dependencies (editable mode):**

   ```bash
   pip install -e .
   ```

## 🧪 Running Tests

To ensure code quality, please run the tests before submitting a pull request.

```bash
pytest
```

## 🔀 Pull Request (PR) Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the `README.md` with details of changes to the interface. This includes new environment variables, exposed ports, useful file locations, and container parameters.
3. Create a new **issue** on the main Aegis repository to discuss the bug or feature you want to work on.
4. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

5. Make your changes and **commit** them with a clear, descriptive message.
6. Push your changes:

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a pull request** from your fork to the main Aegis repository and link it to the issue you created. We will review your PR as soon as possible.

---

**Thank you for your contribution!**
