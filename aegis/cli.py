# aegis/cli.py

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from aegis.core.library import PromptLibrary
from aegis.core.connectors import GeminiConnector, OpenRouterConnector, ModelConnector
from aegis.core.models import ModelResponse, AnalysisResult
from aegis.core.analyzer import LLMAnalyzer # <-- Import the new LLMAnalyzer

app = typer.Typer(
    name="aegis",
    help="Aegis: LLM Red Teaming and Evaluation Framework",
    add_completion=False,
)
console = Console()

# --- Helper Functions (get_connector, display_single_result) remain the same ---
def get_connector(model_identifier: str) -> ModelConnector:
    parts = model_identifier.lower().split('/')
    provider = parts[0]
    try:
        if provider == "gemini":
            return GeminiConnector()
        elif provider == "openrouter":
            if len(parts) < 2:
                console.print("[bold red]Error: OpenRouter model must be specified as 'openrouter/model-name'.[/bold red]")
                raise typer.Exit(code=1)
            model_name = "/".join(parts[1:])
            return OpenRouterConnector(model_name=model_name)
        else:
            console.print(f"[bold red]Error: Provider '{provider}' is not supported.[/bold red]")
            raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Initialization Error for {provider}: {e}[/bold red]")
        raise typer.Exit(code=1)

def display_single_result(response: ModelResponse, analysis: AnalysisResult):
    table = Table(title="Aegis Evaluation Result", show_header=True, header_style="bold magenta")
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

# --- CLI Commands ---

@app.command()
def evaluate(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run."),
    model: str = typer.Option("gemini", "--model", "-m", help="Model to evaluate (e.g., 'gemini', 'openrouter/google/gemini-flash-1.5')."),
):
    """Run a single adversarial prompt evaluation against a specified model."""
    console.print(f"[bold cyan]🚀 Starting Aegis Evaluation...[/bold cyan]")
    library, analyzer = PromptLibrary(), LLMAnalyzer() # Use LLMAnalyzer here too
    library.load_prompts()
    target_prompt = next((p for p in library.get_all() if p.id == prompt_id), None)
    if not target_prompt:
        console.print(f"[bold red]Error: Prompt with ID '{prompt_id}' not found.[/bold red]")
        raise typer.Exit(code=1)
    console.print(f"✅ Found Prompt [bold]'{prompt_id}'[/bold].")
    console.print(f"✅ Initializing and sending to [bold]'{model}'[/bold]...")
    connector = get_connector(model)
    response = connector.send_prompt(target_prompt)
    console.print("✅ Response received.")
    console.print("✅ Analyzing response with LLM evaluator...")
    analysis_result = analyzer.analyze(response, target_prompt)
    console.print("✅ Analysis complete.")
    display_single_result(response, analysis_result)

@app.command(name="batch-evaluate")
def batch_evaluate(
    category: str = typer.Option(..., "--category", "-c", help="The category of prompts to evaluate."),
    model: str = typer.Option("gemini", "--model", "-m", help="Model to evaluate (e.g., 'gemini', 'openrouter/google/gemini-flash-1.5')."),
):
    """Run a batch evaluation for all prompts in a specific category."""
    console.print(f"[bold cyan]🚀 Starting Batch Evaluation for category '{category}'...[/bold cyan]")
    
    library, analyzer = PromptLibrary(), LLMAnalyzer() # <-- Use the new LLMAnalyzer
    library.load_prompts()
    
    prompts_to_run = library.filter_by_category(category)
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
            results.append((prompt.id, analysis, response.output_text)) # <-- Store output_text
            progress.update(task, advance=1, description=f"[green]Evaluated {prompt.id}")

    # --- Display Upgraded Summary Table ---
    summary_table = Table(title=f"Batch Evaluation Summary: {category}", show_header=True, header_style="bold blue")
    summary_table.add_column("Prompt ID", style="dim", width=10)
    summary_table.add_column("Classification", style="bold", width=20)
    summary_table.add_column("Score", width=10)
    summary_table.add_column("Model Output", no_wrap=False) # <-- New column

    total_score = 0
    for prompt_id, analysis, output_text in results:
        total_score += analysis.vulnerability_score
        color = "white"
        if analysis.classification.name == "NON_COMPLIANT": color = "red"
        elif analysis.classification.name == "COMPLIANT": color = "green"
        elif analysis.classification.name == "PARTIAL_COMPLIANCE": color = "yellow"
        summary_table.add_row(
            prompt_id,
            f"[{color}]{analysis.classification.name}[/{color}]",
            f"{analysis.vulnerability_score:.1f}",
            output_text # <-- Display the model's raw output
        )
    
    console.print(summary_table)
    average_score = total_score / len(results) if results else 0
    console.print(Panel(f"Average Vulnerability Score for this category: [bold cyan]{average_score:.2f}[/bold cyan]", title="Overall Result"))


@app.command()
def report():
    """(Placeholder) Generate a report from evaluation results."""
    console.print("[bold yellow]🚧 Report command is under construction.[/bold yellow]")

    