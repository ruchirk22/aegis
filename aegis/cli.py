# aegis/cli.py

import typer
from rich.console import Console
from rich.table import Table

from aegis.core.library import PromptLibrary
from aegis.core.connectors import GeminiConnector, OpenAIConnector, ModelConnector
from aegis.core.models import ModelResponse

# This is the single source of truth for the Typer application.
app = typer.Typer(
    name="aegis",
    help="Aegis: LLM Red Teaming and Evaluation Framework",
    add_completion=False,
)
console = Console()

# --- Helper Functions ---

MODEL_CONNECTORS = {
    "gemini": GeminiConnector,
    "openai": OpenAIConnector,
}

def get_connector(model_name: str) -> ModelConnector:
    """Factory function to get the correct model connector instance."""
    connector_class = MODEL_CONNECTORS.get(model_name.lower())
    if not connector_class:
        console.print(f"[bold red]Error: Model '{model_name}' is not supported.[/bold red]")
        raise typer.Exit(code=1)
    try:
        return connector_class()
    except ValueError as e:
        console.print(f"[bold red]Initialization Error for {model_name}: {e}[/bold red]")
        raise typer.Exit(code=1)

def display_response(response: ModelResponse):
    """Displays the model's response in a formatted table."""
    table = Table(title="LLM Evaluation Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="dim", width=20)
    table.add_column("Value")

    table.add_row("Prompt ID", response.prompt_id)
    table.add_row("Model Name", response.model_name)
    
    if response.error:
        table.add_row("[bold red]Error[/bold red]", response.error)
    else:
        table.add_row("Output Text", response.output_text)
        table.add_row("Metadata", str(response.metadata))

    console.print(table)


# --- CLI Commands ---

@app.command()
def evaluate(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run."),
    model: str = typer.Option("gemini", "--model", "-m", help="The model to evaluate (e.g., 'gemini', 'openai')."),
):
    """
    Run a single adversarial prompt evaluation against a specified model.
    """
    console.print(f"[bold cyan]🚀 Starting Aegis Evaluation...[/bold cyan]")
    library = PromptLibrary()
    library.load_prompts()
    
    target_prompt = next((p for p in library.get_all() if p.id == prompt_id), None)
    if not target_prompt:
        console.print(f"[bold red]Error: Prompt with ID '{prompt_id}' not found in the library.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"✅ Found Prompt [bold]'{prompt_id}'[/bold].")
    console.print(f"✅ Initializing model connector for [bold]'{model}'[/bold]...")
    
    connector = get_connector(model)
    
    console.print(f"✅ Sending prompt to model. Please wait...")
    response = connector.send_prompt(target_prompt)
    
    display_response(response)

@app.command()
def report():
    """
    (Placeholder) Generate a report from evaluation results.
    """
    console.print("[bold yellow]🚧 Report command is under construction.[/bold yellow]")

