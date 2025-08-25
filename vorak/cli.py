# vorak-FRAMEWORK/vorak/cli.py

import typer
import json
import csv
import os
from typing import Optional, List, Dict, Any
from io import BytesIO

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import plotly.express as px
import pandas as pd

from vorak.core.prompt_manager import PromptManager
from vorak.core.connectors import (
    OpenRouterConnector, 
    ModelConnector, 
    UserProvidedGeminiConnector,
    LocalModelConnector,
    OpenAIConnector,
    AnthropicConnector,
    CustomEndpointConnector
)
from vorak.core.models import ModelResponse, AnalysisResult, AdversarialPrompt
from vorak.core.analyzer import LLMAnalyzer
from vorak.core.reporting import generate_pdf_report
from vorak.core.prompt_generator import PromptGenerator
# --- Feature 4: New import for Agent Testing ---
from vorak.agents.tester import AgentTester


app = typer.Typer(
    name="vorak",
    help="vorak: Secure Evaluation of Neural Testing & Red-teaming",
    add_completion=False,
)
console = Console()

def get_connector(model_identifier: str) -> ModelConnector:
    """Helper function to instantiate the correct model connector."""
    # --- Feature 3: Check if the identifier is a local directory path ---
    if os.path.isdir(model_identifier):
        console.print(f"[cyan]Detected local model path: '{model_identifier}'[/cyan]")
        try:
            return LocalModelConnector(model_name=model_identifier)
        except (ImportError, ValueError) as e:
            console.print(f"[bold red]Local Model Error: {e}[/bold red]")
            raise typer.Exit(code=1)

    # --- Existing logic for API-based models ---
    parts = model_identifier.lower().split('/')
    provider = parts[0]
    model_name_only = "/".join(parts[1:])

    try:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be set in your environment.")
            # Default to a good model if not specified
            model_to_use = model_name_only if model_name_only else "gemini-1.5-flash-latest"
            return UserProvidedGeminiConnector(model_name=model_to_use, api_key=api_key)
            
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set in your environment.")
            if not model_name_only:
                raise ValueError("OpenAI model name must be specified (e.g., 'openai/gpt-4o-mini').")
            return OpenAIConnector(model_name=model_name_only, api_key=api_key)

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set in your environment.")
            if not model_name_only:
                raise ValueError("Anthropic model name must be specified (e.g., 'anthropic/claude-3-5-sonnet-20240620').")
            return AnthropicConnector(model_name=model_name_only, api_key=api_key)

        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY must be set in your environment.")
            if not model_name_only:
                raise ValueError("OpenRouter model name must be specified (e.g., 'openrouter/google/gemma-2-9b-it').")
            return OpenRouterConnector(model_name=model_name_only, api_key=api_key)
            
        else:
            console.print(f"[bold red]Error: Provider '{provider}' is not supported. For local models, provide a valid directory path.[/bold red]")
            raise typer.Exit(code=1)
            
    except ValueError as e:
        console.print(f"[bold red]Initialization Error for {provider}: {e}[/bold red]")
        raise typer.Exit(code=1)

def display_single_result(response: ModelResponse, analysis: AnalysisResult):
    """Displays a single evaluation result in a formatted table and panels."""
    table = Table(title="vorak Evaluation Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="dim", width=20)
    table.add_column("Value")
    color = "white"
    if analysis.classification.name == "NON_COMPLIANT": color = "bold red"
    elif analysis.classification.name == "COMPLIANT": color = "bold green"
    elif analysis.classification.name == "PARTIAL_COMPLIANCE": color = "bold yellow"
    table.add_row("Prompt ID", response.prompt_id)
    table.add_row("Model Name", response.model_name)
    table.add_row(f"[{color}]Classification[/{color}]", f"[{color}]{analysis.classification.name}[/{color}]")
    table.add_row("Vulnerability Score", f"{analysis.vulnerability_score:.1f} / 100.0")
    console.print(table)
    console.print(Panel(response.output_text, title="[cyan]Model Output[/cyan]", border_style="cyan"))
    console.print(Panel(analysis.explanation, title="[cyan]Analysis Explanation[/cyan]", border_style="cyan"))

def save_results_to_json(results: List[Dict[str, Any]], filepath: str):
    """Saves a list of result dictionaries to a JSON file."""
    export_data = []
    for result in results:
        export_data.append({
            "prompt_id": result["prompt"].id,
            "category": result["prompt"].category,
            "prompt_text": result["prompt"].prompt_text,
            "model_name": result["response"].model_name,
            "model_output": result["response"].output_text,
            "classification": result["analysis"].classification.name,
            "vulnerability_score": result["analysis"].vulnerability_score,
            "explanation": result["analysis"].explanation,
        })
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

def save_results_to_csv(results: List[Dict[str, Any]], filepath: str):
    """Saves a list of result dictionaries to a CSV file."""
    headers = [
        "prompt_id", "category", "prompt_text", "model_name", "model_output",
        "classification", "vulnerability_score", "explanation"
    ]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "prompt_id": result["prompt"].id,
                "category": result["prompt"].category,
                "prompt_text": result["prompt"].prompt_text,
                "model_name": result["response"].model_name,
                "model_output": result["response"].output_text,
                "classification": result["analysis"].classification.name,
                "vulnerability_score": result["analysis"].vulnerability_score,
                "explanation": result["analysis"].explanation,
            })

@app.command()
def evaluate(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run."),
    model: str = typer.Option(..., "--model", "-m", help="Model to evaluate (e.g., 'gemini/gemini-1.5-flash-latest', 'openai/gpt-4o-mini', or a local path like './models/my-llama')."),
):
    """Run a single adversarial prompt evaluation against a specified model."""
    console.print(f"[bold cyan]🚀 Starting vorak Evaluation...[/bold cyan]")
    manager, analyzer = PromptManager(), LLMAnalyzer()
    manager.load_prompts()
    target_prompt = next((p for p in manager.get_all() if p.id == prompt_id), None)
    if not target_prompt:
        console.print(f"[bold red]Error: Prompt with ID '{prompt_id}' not found.[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"✅ Found Prompt [bold]'{prompt_id}'[/bold].")
    console.print(f"✅ Initializing and sending to [bold]'{model}'[/bold]...")
    connector = get_connector(model)
    response = connector.send_prompt(target_prompt)
    console.print("✅ Response received.")
    console.print("✅ Analyzing response with evaluators...")
    analysis_result = analyzer.analyze(response, target_prompt)
    console.print("✅ Analysis complete.")
    display_single_result(response, analysis_result)


@app.command(name="batch-evaluate")
def batch_evaluate(
    category: str = typer.Option(..., "--category", "-c", help="The category of prompts to evaluate."),
    model: str = typer.Option(..., "--model", "-m", help="Model to evaluate (e.g., 'gemini/gemini-1.5-flash-latest', or a local path)."),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Path to save results as a JSON file."),
    output_csv: Optional[str] = typer.Option(None, "--output-csv", help="Path to save results as a CSV file."),
    output_pdf: Optional[str] = typer.Option(None, "--output-pdf", help="Path to save a PDF report."),
):
    """Run a batch evaluation for a category and optionally save the results."""
    console.print(f"[bold cyan]🚀 Starting Batch Evaluation for category '{category}'...[/bold cyan]")
    manager, analyzer = PromptManager(), LLMAnalyzer()
    manager.load_prompts()
    prompts_to_run = manager.filter_by_category(category)
    if not prompts_to_run:
        console.print(f"[bold red]Error: No prompts found for category '{category}'.[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"✅ Found {len(prompts_to_run)} prompts. Initializing connector for [bold]'{model}'[/bold]...")
    connector = get_connector(model)
    results = []
    with Progress() as progress:
        task = progress.add_task(f"[green]Evaluating {category}...", total=len(prompts_to_run))
        for prompt in prompts_to_run:
            response = connector.send_prompt(prompt)
            analysis = analyzer.analyze(response, prompt)
            results.append({"prompt": prompt, "response": response, "analysis": analysis})
            progress.update(task, advance=1, description=f"[green]Evaluated {prompt.id}")

    summary_table = Table(title=f"Batch Evaluation Summary: {category}", show_header=True, header_style="bold blue")
    summary_table.add_column("Prompt ID", style="dim", width=10)
    summary_table.add_column("Classification", style="bold", width=20)
    summary_table.add_column("Score", width=10)
    summary_table.add_column("Model Output", no_wrap=False)
    total_score = 0
    classifications = []
    for result in results:
        total_score += result["analysis"].vulnerability_score
        classifications.append(result["analysis"].classification.name)
        color = "white"
        if result["analysis"].classification.name == "NON_COMPLIANT": color = "red"
        elif result["analysis"].classification.name == "COMPLIANT": color = "green"
        elif result["analysis"].classification.name == "PARTIAL_COMPLIANCE": color = "yellow"
        summary_table.add_row(
            result["prompt"].id,
            f"[{color}]{result['analysis'].classification.name}[/{color}]",
            f"{result['analysis'].vulnerability_score:.1f}",
            result["response"].output_text
        )
    console.print(summary_table)
    average_score = total_score / len(results) if results else 0
    console.print(Panel(f"Average Vulnerability Score for this category: [bold cyan]{average_score:.2f}[/bold cyan]", title="Overall Result"))
    if output_json:
        save_results_to_json(results, output_json)
        console.print(f"✅ Full results saved to [bold green]{output_json}[/bold green]")
    if output_csv:
        save_results_to_csv(results, output_csv)
        console.print(f"✅ Full results saved to [bold green]{output_csv}[/bold green]")
    if output_pdf:
        console.print(f"🎨 Generating PDF report...")
        try:
            chart_image_buffer = BytesIO()
            classification_counts = pd.Series(classifications).value_counts()
            fig_bar = px.bar(
                classification_counts, x=classification_counts.index, y=classification_counts.values,
                labels={'x': 'Classification', 'y': 'Count'}, title="Classification Breakdown",
                color=classification_counts.index,
                color_discrete_map={
                    'NON_COMPLIANT': 'red', 'COMPLIANT': 'green', 'PARTIAL_COMPLIANCE': 'orange',
                    'AMBIGUOUS': 'grey', 'ERROR': 'black'
                }
            )
            fig_bar.write_image(chart_image_buffer, format='png')
            chart_image_buffer.seek(0)

            pdf_buffer = BytesIO()
            generate_pdf_report(results, pdf_buffer, chart_image_buffer)
            
            with open(output_pdf, "wb") as f:
                f.write(pdf_buffer.getvalue())

            console.print(f"✅ PDF report saved to [bold green]{output_pdf}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error generating PDF report: {e}[/bold red]")
            console.print("[bold yellow]Please ensure 'kaleido' is installed (`pip install kaleido`)[/bold yellow]")

# --- Feature 4: New CLI command for Agent Testing ---
@app.command(name="evaluate-agent")
def evaluate_agent(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run against the agent."),
):
    """Run a single evaluation against a LangChain agent."""
    console.print(f"[bold cyan]🤖 Starting vorak Agent Evaluation...[/bold cyan]")
    
    try:
        agent_tester = AgentTester()
    except (ImportError, ValueError) as e:
        console.print(f"[bold red]Agent Initialization Error: {e}[/bold red]")
        raise typer.Exit(code=1)

    manager, analyzer = PromptManager(), LLMAnalyzer()
    manager.load_prompts()
    target_prompt = next((p for p in manager.get_all() if p.id == prompt_id), None)
    
    if not target_prompt:
        console.print(f"[bold red]Error: Prompt with ID '{prompt_id}' not found.[/bold red]")
        raise typer.Exit(code=1)
        
    console.print(f"✅ Found Prompt [bold]'{prompt_id}'[/bold].")
    console.print("✅ Sending prompt to agent...")
    
    response = agent_tester.evaluate_agent(target_prompt)
    
    console.print("✅ Response received from agent.")
    console.print("✅ Analyzing agent response...")
    
    analysis_result = analyzer.analyze(response, target_prompt)
    
    console.print("✅ Analysis complete.")
    display_single_result(response, analysis_result)

@app.command(name="add-prompt")
def add_prompt(
    category: str = typer.Option(..., "--category", "-c", help="Category of the prompt (e.g., 'Jailbreaking_Role-Playing')."),
    severity: str = typer.Option(..., "--severity", "-s", help="Severity of the prompt (e.g., 'HIGH', 'CRITICAL')."),
    prompt_text: str = typer.Option(..., "--prompt", "-p", help="The full text of the adversarial prompt."),
    id: Optional[str] = typer.Option(None, "--id", help="A unique ID for the prompt. If not provided, one will be generated."),
):
    """Add a new adversarial prompt to the central prompt library."""
    console.print("[bold cyan]📝 Adding new prompt to the library...[/bold cyan]")
    manager = PromptManager()
    if not id:
        cat_prefix = "".join([word[0] for word in category.split('_')]).upper()
        new_id_num = len(manager.get_all()) + 1
        id = f"{cat_prefix}_{new_id_num:03d}"
        console.print(f"Generated new prompt ID: [bold yellow]{id}[/bold yellow]")
    new_prompt = AdversarialPrompt(
        id=id, category=category, subcategory="Custom_CLI_Entry",
        severity=severity.upper(), prompt_text=prompt_text, expected_behavior="REJECT",
    )
    if manager.add_prompt(new_prompt, save=True):
        console.print(f"✅ [bold green]Successfully added and saved prompt '{id}' to the library.[/bold green]")
    else:
        console.print(f"❌ [bold red]Failed to add prompt. An entry with ID '{id}' may already exist.[/bold red]")

@app.command(name="generate-prompts")
def generate_prompts(
    category: str = typer.Option(..., "--category", "-c", help="The base category to generate prompts from."),
    num: int = typer.Option(10, "--num", "-n", help="Number of prompts to generate."),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="JSON file to save the generated prompts. If not provided, prompts are added to the main library."),
):
    """Generate new adversarial prompts using augmentation techniques."""
    console.print(f"[bold cyan]🧬 Starting prompt generation...[/bold cyan]")
    generator = PromptGenerator()
    new_prompts = generator.generate_prompts(base_category=category, num_to_generate=num)
    if not new_prompts:
        console.print("[bold red]❌ Prompt generation failed. See errors above.[/bold red]")
        raise typer.Exit(code=1)
    if output_file:
        prompts_as_dicts = [p.to_dict() for p in new_prompts]
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(prompts_as_dicts, f, indent=2)
            console.print(f"✅ [bold green]Successfully saved {len(new_prompts)} new prompts to '{output_file}'.[/bold green]")
        except IOError as e:
            console.print(f"[bold red]❌ Error writing to file '{output_file}': {e}[/bold red]")
    else:
        console.print("Adding generated prompts to the main library...")
        manager = PromptManager()
        count = 0
        for prompt in new_prompts:
            if manager.add_prompt(prompt, save=False):
                count += 1
        manager.save_library()
        console.print(f"✅ [bold green]Successfully added {count} new prompts to the main library.[/bold green]")


if __name__ == "__main__":
    app()