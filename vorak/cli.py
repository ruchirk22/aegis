# vorak/cli.py

import typer
import json
import csv
import os
from typing import Optional, List, Dict, Any
from io import BytesIO
from enum import Enum
from datetime import datetime
import uuid
import yaml

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import plotly.express as px
import pandas as pd
from rich.text import Text
from rich.tree import Tree
from rich.markdown import Markdown

from vorak.core.prompt_manager import PromptManager
from vorak.core.connectors import (
    OpenRouterConnector, ModelConnector, UserProvidedGeminiConnector,
    LocalModelConnector, OpenAIConnector, AnthropicConnector, CustomEndpointConnector
)
from vorak.core.models import (
    ModelResponse, AnalysisResult, AdversarialPrompt, EvaluationMode,
    Classification, GovernanceResult, PromptGenerationStrategy
)
from vorak.core.analyzer import LLMAnalyzer
# --- MODIFIED: Import DatabaseManager ---
from vorak.core.database.manager import DatabaseManager
from vorak.core.reporting import generate_pdf_report, generate_comparison_pdf_report
from vorak.core.comparison import ComparisonReport
from vorak.core.prompt_generator import PromptGenerator
from vorak.agents.tester import AgentTester

app = typer.Typer(
    name="vorak",
    help="Vorak: Vulnerability Oriented Red-teaming for AI Knowledge",
    add_completion=False,
)

# --- NEW: Create a new Typer app for the 'prompt' command group ---
prompt_app = typer.Typer(name="prompt", help="Manage and contribute to the prompt library.")
app.add_typer(prompt_app)

console = Console()

def get_connector(model_identifier: str) -> ModelConnector:
    """Helper function to instantiate the correct model connector."""
    if os.path.isdir(model_identifier):
        console.print(f"[cyan]Detected local model path: '{model_identifier}'[/cyan]")
        try:
            return LocalModelConnector(model_name=model_identifier)
        except (ImportError, ValueError) as e:
            console.print(f"[bold red]Local Model Error: {e}[/bold red]")
            raise typer.Exit(code=1)
    parts = model_identifier.lower().split('/')
    provider = parts[0]
    model_name_only = "/".join(parts[1:])
    try:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key: raise ValueError("GEMINI_API_KEY must be set in your environment.")
            model_to_use = model_name_only if model_name_only else "gemini-1.5-flash-latest"
            return UserProvidedGeminiConnector(model_name=model_to_use, api_key=api_key)
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY must be set in your environment.")
            if not model_name_only: raise ValueError("OpenAI model name must be specified.")
            return OpenAIConnector(model_name=model_name_only, api_key=api_key)
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY must be set in your environment.")
            if not model_name_only: raise ValueError("Anthropic model name must be specified.")
            return AnthropicConnector(model_name=model_name_only, api_key=api_key)
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key: raise ValueError("OPENROUTER_API_KEY must be set in your environment.")
            if not model_name_only: raise ValueError("OpenRouter model name must be specified.")
            return OpenRouterConnector(model_name=model_name_only, api_key=api_key)
        else:
            console.print(f"[bold red]Error: Provider '{provider}' is not supported.[/bold red]")
            raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Initialization Error for {provider}: {e}[/bold red]")
        raise typer.Exit(code=1)

# --- NEW: Helper function to flatten results for DB insertion ---
def convert_result_to_flat_dict(result_data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Converts a nested result dictionary into a flat dictionary."""
    gov_data = result_data["analysis"].governance
    return {
        "session_id": session_id,
        "prompt_id": result_data["prompt"].id,
        "category": result_data["prompt"].category,
        "prompt_text": result_data["prompt"].prompt_text,
        "model_name": result_data["response"].model_name,
        "model_output": result_data["response"].output_text,
        "classification": result_data["analysis"].classification.name,
        "vulnerability_score": result_data["analysis"].vulnerability_score,
        "explanation": result_data["analysis"].explanation,
        "governance_nist": ", ".join(gov_data.nist_ai_rmf) if gov_data else "",
        "governance_eu": ", ".join(gov_data.eu_ai_act) if gov_data else "",
        "governance_iso": ", ".join(gov_data.iso_iec_23894) if gov_data else "",
    }

# --- MODIFIED: Added MITRE ATLAS to the governance display ---
def display_governance_risks(governance: Optional[GovernanceResult]):
    """Displays governance and compliance risks in a formatted tree."""
    if not governance:
        return
    
    tree = Tree("[bold bright_blue]🏛️ Governance & Compliance Risks[/bold bright_blue]", guide_style="bright_blue")
    
    if governance.nist_ai_rmf:
        nist_branch = tree.add("[bold]NIST AI RMF[/bold]")
        for item in governance.nist_ai_rmf: nist_branch.add(f"[cyan]{item}")
            
    if governance.eu_ai_act:
        eu_branch = tree.add("[bold]EU AI Act[/bold]")
        for item in governance.eu_ai_act: eu_branch.add(f"[cyan]{item}")

    if governance.iso_iec_23894:
        iso_branch = tree.add("[bold]ISO/IEC 23894[/bold]")
        for item in governance.iso_iec_23894: iso_branch.add(f"[cyan]{item}")

    if governance.mitre_atlas:
        atlas_branch = tree.add("[bold]MITRE ATLAS[/bold]")
        for item in governance.mitre_atlas: atlas_branch.add(f"[cyan]{item}")
            
    console.print(tree)

def display_single_result(response: ModelResponse, analysis: AnalysisResult, title: str = "Vorak Evaluation Result"):
    """Displays a single evaluation result in a formatted table and panels."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
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
    console.print(Panel(Text(response.output_text), title="[cyan]Model Output[/cyan]", border_style="cyan"))
    console.print(Panel(Text(analysis.explanation), title="[cyan]Analysis Explanation[/cyan]", border_style="cyan"))
    display_governance_risks(analysis.governance)

def save_results_to_json(results: List[Dict[str, Any]], filepath: str, session_id: str):
    """Saves a list of result dictionaries to a JSON file."""
    export_data = [convert_result_to_flat_dict(res, session_id) for res in results]
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

def save_results_to_csv(results: List[Dict[str, Any]], filepath: str, session_id: str):
    """Saves a list of result dictionaries to a CSV file."""
    headers = list(convert_result_to_flat_dict(results[0], session_id).keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow(convert_result_to_flat_dict(result, session_id))

@app.command()
def evaluate(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run."),
    model: str = typer.Option(..., "--model", "-m", help="Model to evaluate (e.g., 'gemini/gemini-1.5-flash-latest')."),
    mode: EvaluationMode = typer.Option(EvaluationMode.STANDARD, "--mode", help="The evaluation mode.", case_sensitive=False),
    turns: int = typer.Option(3, "--turns", "-t", help="Number of turns for a scenario evaluation."),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Path to save results as a JSON file."),
    output_csv: Optional[str] = typer.Option(None, "--output-csv", help="Path to save results as a CSV file."),
    output_pdf: Optional[str] = typer.Option(None, "--output-pdf", help="Path to save a PDF report."),
):
    """Run a single adversarial prompt evaluation against a specified model."""
    console.print(f"[bold cyan]🚀 Starting Vorak Evaluation in '{mode.value}' mode...[/bold cyan]")
    # --- MODIFIED: Initialize DB Manager and create session ID ---
    db_manager = DatabaseManager()
    session_id = f"{mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    console.print(f"Session ID: [bold yellow]{session_id}[/bold yellow]")

    manager, analyzer = PromptManager(), LLMAnalyzer()
    manager.load_prompts()
    target_prompt = next((p for p in manager.get_all() if p.id == prompt_id), None)
    if not target_prompt:
        console.print(f"[bold red]Error: Prompt with ID '{prompt_id}' not found.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"✅ Found Prompt [bold]'{target_prompt.id}'[/bold].")
    console.print(f"✅ Initializing connector for [bold]'{model}'[/bold]...")
    connector = get_connector(model)
    results_for_report = []

    if mode == EvaluationMode.STANDARD or mode == EvaluationMode.GOVERNANCE:
        console.print(Panel(Text(target_prompt.prompt_text), title=f"[yellow]Prompt: {target_prompt.id}[/yellow]", border_style="yellow"))
        response = connector.send_prompt(target_prompt)
        analysis_result = analyzer.analyze(response, target_prompt)
        result_data = {"prompt": target_prompt, "response": response, "analysis": analysis_result}
        results_for_report.append(result_data)
        db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
        display_single_result(response, analysis_result)
    
    elif mode == EvaluationMode.ADAPTIVE:
        console.print(Panel(Text(target_prompt.prompt_text), title=f"[yellow]Initial Prompt: {target_prompt.id}[/yellow]", border_style="yellow"))
        response = connector.send_prompt(target_prompt)
        analysis_result = analyzer.analyze(response, target_prompt)
        result_data = {"prompt": target_prompt, "response": response, "analysis": analysis_result}
        results_for_report.append(result_data)
        db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
        display_single_result(response, analysis_result, title="[bold]Initial Evaluation Result[/bold]")
        if analysis_result.classification == Classification.COMPLIANT:
            console.rule("[bold yellow]Attempting Adaptive Escalation[/bold yellow]")
            new_prompt = analyzer.run_adaptive_escalation(target_prompt, response, attempt_num=1)
            if new_prompt:
                console.print(Panel(Text(new_prompt.prompt_text), title=f"[yellow]Adaptive Prompt: {new_prompt.id}[/yellow]", border_style="yellow"))
                new_response = connector.send_prompt(new_prompt)
                new_analysis = analyzer.analyze(new_response, new_prompt)
                new_result_data = {"prompt": new_prompt, "response": new_response, "analysis": new_analysis}
                results_for_report.append(new_result_data)
                db_manager.insert_result(convert_result_to_flat_dict(new_result_data, session_id))
                display_single_result(new_response, new_analysis, title="[bold]Adaptive Evaluation Result[/bold]")
        else:
            console.print("[bold green]Initial prompt was not compliant. No need for adaptive escalation.[/bold green]")

    elif mode == EvaluationMode.SCENARIO:
        console.print(f"Running in Scenario mode for {turns} turns...")
        conversation_history = []
        current_prompt = target_prompt
        for i in range(turns):
            turn_num = i + 1
            console.rule(f"[bold yellow]Scenario Turn {turn_num}/{turns}[/bold yellow]")
            console.print(Panel(Text(current_prompt.prompt_text), title=f"[yellow]Attacker Prompt (Turn {turn_num})[/yellow]", border_style="yellow"))
            response = connector.send_prompt(current_prompt, conversation_history)
            analysis = analyzer.analyze(response, current_prompt)
            result_data = {"prompt": current_prompt, "response": response, "analysis": analysis}
            results_for_report.append(result_data)
            db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
            display_single_result(response, analysis, title=f"Turn {turn_num} Evaluation Result")
            conversation_history.append({"role": "user", "content": current_prompt.prompt_text})
            conversation_history.append({"role": "assistant", "content": response.output_text})
            if analysis.classification == Classification.NON_COMPLIANT:
                console.print("[bold red]Scenario successful: Model became non-compliant. Ending evaluation.[/bold red]")
                break
            if turn_num < turns:
                current_prompt = analyzer.generate_next_turn(conversation_history, target_prompt, turn_num + 1)
                if not current_prompt:
                    console.print("[bold red]Could not generate next scenario prompt. Ending evaluation.[/bold red]")
                    break

    elif mode == EvaluationMode.ATTACK_ONLY:
        console.print("Running in Attack-Only mode...")
        response = connector.send_prompt(target_prompt)
        console.print("✅ Response received. Skipping analysis.")
        console.print(Panel(response.output_text, title="[cyan]Model Output[/cyan]", border_style="cyan"))
        raise typer.Exit()

    elif mode == EvaluationMode.ANALYSIS_ONLY:
        console.print("[bold yellow]🚧 Analysis-Only mode is not yet implemented.[/bold yellow]")
        raise typer.Exit()
        
    else:
        console.print(f"[bold red]Error: Mode '{mode.value}' is not recognized or implemented.[/bold red]")
        raise typer.Exit(code=1)

    if results_for_report and (output_json or output_csv or output_pdf):
        if output_json:
            save_results_to_json(results_for_report, output_json, session_id)
            console.print(f"✅ Full results saved to [bold green]{output_json}[/bold green]")
        if output_csv:
            save_results_to_csv(results_for_report, output_csv, session_id)
            console.print(f"✅ Full results saved to [bold green]{output_csv}[/bold green]")
        if output_pdf:
            console.print(f"🎨 Generating PDF report...")
            try:
                classifications = [res["analysis"].classification.name for res in results_for_report]
                chart_image_buffer = BytesIO()
                classification_counts = pd.Series(classifications).value_counts()
                fig_bar = px.bar(
                    classification_counts, x=classification_counts.index, y=classification_counts.values,
                    labels={'x': 'Classification', 'y': 'Count'}, title="Classification Breakdown",
                    color=classification_counts.index,
                    color_discrete_map={'NON_COMPLIANT': 'red', 'COMPLIANT': 'green', 'PARTIAL_COMPLIANCE': 'orange', 'AMBIGUOUS': 'grey', 'ERROR': 'black'}
                )
                fig_bar.write_image(chart_image_buffer, format='png', engine='kaleido')
                chart_image_buffer.seek(0)
                pdf_buffer = BytesIO()
                generate_pdf_report(results_for_report, pdf_buffer, chart_image_buffer)
                with open(output_pdf, "wb") as f: f.write(pdf_buffer.getvalue())
                console.print(f"✅ PDF report saved to [bold green]{output_pdf}[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Error generating PDF report: {e}[/bold red]")
                console.print("[bold yellow]Please ensure 'kaleido' is installed (`pip install kaleido`)[/bold yellow]")

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
    db_manager = DatabaseManager()
    session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    console.print(f"Session ID: [bold yellow]{session_id}[/bold yellow]")

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
            result_data = {"prompt": prompt, "response": response, "analysis": analysis}
            results.append(result_data)
            db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
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
        save_results_to_json(results, output_json, session_id)
        console.print(f"✅ Full results saved to [bold green]{output_json}[/bold green]")
    if output_csv:
        save_results_to_csv(results, output_csv, session_id)
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
                color_discrete_map={'NON_COMPLIANT': 'red', 'COMPLIANT': 'green', 'PARTIAL_COMPLIANCE': 'orange', 'AMBIGUOUS': 'grey', 'ERROR': 'black'}
            )
            fig_bar.write_image(chart_image_buffer, format='png', engine='kaleido')
            chart_image_buffer.seek(0)
            pdf_buffer = BytesIO()
            generate_pdf_report(results, pdf_buffer, chart_image_buffer)
            with open(output_pdf, "wb") as f: f.write(pdf_buffer.getvalue())
            console.print(f"✅ PDF report saved to [bold green]{output_pdf}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error generating PDF report: {e}[/bold red]")
            console.print("[bold yellow]Please ensure 'kaleido' is installed (`pip install kaleido`)[/bold yellow]")

@app.command(name="compare-sessions")
def compare_sessions(
    session_a_id: str = typer.Argument(..., help="The baseline session ID (Session A)."),
    session_b_id: str = typer.Argument(..., help="The candidate session ID to compare (Session B)."),
    output_pdf: Optional[str] = typer.Option(None, "--output-pdf", help="Path to save a PDF comparison report."),
):
    """Compare the results of two evaluation sessions."""
    console.print(f"[bold cyan]📊 Comparing Session A ('{session_a_id}') vs. Session B ('{session_b_id}')...[/bold cyan]")
    
    try:
        db_manager = DatabaseManager()
        report = ComparisonReport(session_a_id, session_b_id, db_manager)
        summary = report.summary
        
        summary_panel = Panel(
            f"[bold]Avg. Score (A):[/bold] {summary.avg_score_a:.2f}\n"
            f"[bold]Avg. Score (B):[/bold] {summary.avg_score_b:.2f}\n"
            f"[bold]Overall Delta:[/bold] {summary.avg_score_delta:+.2f}\n\n"
            f"[green]Improvements:[/green] {summary.improvements}\n"
            f"[red]Regressions:[/red] {summary.regressions}\n"
            f"[dim]Unchanged:[/dim] {summary.unchanged}",
            title="[bold]Comparison Summary[/bold]", expand=False
        )
        console.print(summary_panel)

        table = Table(title="Detailed Comparison", show_header=True, header_style="bold blue")
        table.add_column("Prompt ID", style="dim")
        table.add_column("Score (A)")
        table.add_column("Score (B)")
        table.add_column("Delta")
        table.add_column("Status")
        for res in report.results:
            delta_str = f"{res.delta:+.1f}"
            status_color = "white"
            if res.status == "Improvement": status_color = "green"
            elif res.status == "Regression": status_color = "red"
            table.add_row(res.prompt_id, f"{res.score_a:.1f}", f"{res.score_b:.1f}", f"[{status_color}]{delta_str}[/{status_color}]", f"[{status_color}]{res.status}[/{status_color}]")
        console.print(table)

        if output_pdf:
            console.print(f"🎨 Generating PDF comparison report...")
            chart_data = {'Status': ['Improvements', 'Regressions', 'Unchanged'], 'Count': [summary.improvements, summary.regressions, summary.unchanged]}
            fig = px.bar(pd.DataFrame(chart_data), x='Status', y='Count', color='Status', color_discrete_map={'Improvements': 'green', 'Regressions': 'red', 'Unchanged': 'grey'}, title='Comparison Breakdown')
            chart_buffer = BytesIO()
            fig.write_image(chart_buffer, format='png', engine='kaleido')
            pdf_buffer = BytesIO()
            generate_comparison_pdf_report(report, pdf_buffer, chart_buffer)
            with open(output_pdf, "wb") as f: f.write(pdf_buffer.getvalue())
            console.print(f"✅ Comparison report saved to [bold green]{output_pdf}[/bold green]")

    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]Error generating comparison: {e}[/bold red]")
        raise typer.Exit(code=1)
    except ImportError:
        console.print("[bold red]Error: Please install pandas to use the compare feature (`pip install pandas`)[/bold red]")
        raise typer.Exit(code=1)

@app.command(name="evaluate-agent")
def evaluate_agent(
    prompt_id: str = typer.Option(..., "--prompt-id", "-p", help="The ID of the prompt to run against the agent."),
):
    """Run a single evaluation against a LangChain agent."""
    console.print(f"[bold cyan]🤖 Starting Vorak Agent Evaluation...[/bold cyan]")
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

# --- MODIFIED: Update the generate-prompts command ---
@app.command(name="generate-prompts")
def generate_prompts(
    category: str = typer.Option(..., "--category", "-c", help="The base category to generate prompts from."),
    num: int = typer.Option(10, "--num", "-n", help="Number of prompts to generate."),
    strategy: PromptGenerationStrategy = typer.Option(
        PromptGenerationStrategy.SYNONYM,
        "--strategy",
        "-s",
        help="The augmentation strategy to use.",
        case_sensitive=False,
    ),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="JSON file to save the generated prompts."),
):
    """Generate new adversarial prompts using augmentation techniques."""
    console.print(f"[bold cyan]🧬 Starting prompt generation with '{strategy.value}' strategy...[/bold cyan]")
    generator = PromptGenerator()
    new_prompts = generator.generate_prompts(base_category=category, num_to_generate=num, strategy=strategy.value)
    
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


# --- NEW: Feature 1 - Community Contribution Command ---
@prompt_app.command("contribute")
def contribute_prompts(
    file: str = typer.Option(..., "--file", "-f", help="Path to a JSON file containing new prompts to contribute."),
):
    """Validate and prepare new prompts for community contribution."""
    console.print(f"[bold cyan]🔍 Validating contribution file: '{file}'...[/bold cyan]")
    
    if not os.path.exists(file):
        console.print(f"[bold red]Error: File not found at '{file}'.[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(file, 'r', encoding='utf-8') as f:
            new_prompts_data = json.load(f)
    except json.JSONDecodeError:
        console.print(f"[bold red]Error: Invalid JSON format in '{file}'.[/bold red]")
        raise typer.Exit(code=1)

    if not isinstance(new_prompts_data, list):
        console.print(f"[bold red]Error: The JSON file must contain a list of prompt objects.[/bold red]")
        raise typer.Exit(code=1)

    manager = PromptManager()
    validated_prompts = []
    errors = []

    for i, data in enumerate(new_prompts_data):
        try:
            # Validate structure by attempting to create the dataclass
            prompt = AdversarialPrompt(**data)
            
            # Check for duplicate IDs
            if manager.id_exists(prompt.id):
                errors.append(f"Prompt {i+1} (ID: '{prompt.id}') failed validation: ID already exists in the library.")
            else:
                validated_prompts.append(prompt)
        except TypeError as e:
            # This catches missing keys or incorrect field names
            errors.append(f"Prompt {i+1} (ID: '{data.get('id', 'N/A')}') failed validation: {e}")

    if errors:
        console.print("[bold red]Validation failed. Please fix the following errors:[/bold red]")
        for error in errors:
            console.print(f"- {error}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]✅ Validation successful! {len(validated_prompts)} new prompts are ready for contribution.[/bold green]")
    console.print("\n[bold]Please copy the text below and paste it into a new GitHub issue or pull request:[/bold]")
    
    # Generate formatted output for GitHub
    github_body = f"### New Prompt Contribution\n\n**Number of new prompts:** {len(validated_prompts)}\n\n"
    github_body += "--- \n\n"
    for prompt in validated_prompts:
        github_body += f"**ID:** `{prompt.id}`\n"
        github_body += f"**Category:** {prompt.category}\n"
        github_body += f"**Severity:** {prompt.severity}\n"
        github_body += "**Prompt Text:**\n"
        github_body += f"```\n{prompt.prompt_text}\n```\n"
        github_body += "---\n"
        
    console.print(Panel(Markdown(github_body), title="[bold]Contribution Body[/bold]", border_style="yellow"))

# --- NEW: Command to clean up the prompt library ---
@prompt_app.command("cleanup")
def cleanup_library():
    """Removes unused fields from the main prompt_library.json file."""
    console.print("[bold cyan]🧹 Cleaning up the prompt library...[/bold cyan]")
    manager = PromptManager()
    
    # The to_dict() method in our AdversarialPrompt model already
    # returns only the fields we care about. We just need to load
    # and re-save the library.
    
    # This forces a load from the file
    all_prompts = manager.get_all()
    
    if not all_prompts:
        console.print("[bold red]Error: Could not load any prompts from the library.[/bold red]")
        raise typer.Exit(code=1)
        
    # The save_library method will write the cleaned data back to the file
    manager.save_library()
    
    console.print("[bold green]✅ Prompt library has been successfully cleaned and formatted![/bold green]")

# --- NEW: Command to clean up the prompt library ---
@prompt_app.command("cleanup")
def cleanup_library():
    """Removes unused fields from the main prompt_library.json file."""
    console.print("[bold cyan]🧹 Cleaning up the prompt library...[/bold cyan]")
    manager = PromptManager()
    
    # The to_dict() method in our AdversarialPrompt model already
    # returns only the fields we care about. We just need to load
    # and re-save the library.
    
    # This forces a load from the file
    all_prompts = manager.get_all()
    
    if not all_prompts:
        console.print("[bold red]Error: Could not load any prompts from the library.[/bold red]")
        raise typer.Exit(code=1)
        
    # The save_library method will write the cleaned data back to the file
    manager.save_library()
    
    console.print("[bold green]✅ Prompt library has been successfully cleaned and formatted![/bold green]")

@app.command("run")
def run_playbook(
    playbook_file: str = typer.Option(..., "--playbook", "-p", help="Path to the YAML playbook file to execute."),
):
    """Execute a multi-step testing plan from a YAML playbook."""
    console.print(f"[bold cyan]▶️  Executing playbook: '{playbook_file}'[/bold cyan]")

    if not os.path.exists(playbook_file):
        console.print(f"[bold red]Error: Playbook file not found at '{playbook_file}'.[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(playbook_file, 'r', encoding='utf-8') as f:
            playbook = yaml.safe_load(f)
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing YAML file: {e}[/bold red]")
        raise typer.Exit(code=1)

    if not isinstance(playbook, list):
        console.print("[bold red]Error: Playbook must be a list of steps.[/bold red]")
        raise typer.Exit(code=1)
    
    # Instantiate core components once
    db_manager = DatabaseManager()
    analyzer = LLMAnalyzer()
    prompt_manager = PromptManager()

    for i, step in enumerate(playbook):
        step_name = step.get("name", f"Step {i+1}")
        command = step.get("command")
        params = step.get("params", {})
        
        console.rule(f"[bold]Executing: {step_name}[/bold]")

        if not command or not isinstance(params, dict):
            console.print(f"[bold red]Skipping invalid step '{step_name}': Missing 'command' or 'params'.[/bold red]")
            continue

        try:
            # We call the underlying logic directly instead of using typer.invoke
            if command == "evaluate":
                # Re-implement the core logic of the 'evaluate' command
                session_id = f"playbook_evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                console.print(f"Session ID: [bold yellow]{session_id}[/bold yellow]")
                connector = get_connector(params["model"])
                prompt = next((p for p in prompt_manager.get_all() if p.id == params["prompt_id"]), None)
                if not prompt:
                    console.print(f"[bold red]Error in step '{step_name}': Prompt ID '{params['prompt_id']}' not found.[/bold red]")
                    continue
                
                response = connector.send_prompt(prompt)
                analysis = analyzer.analyze(response, prompt)
                result_data = {"prompt": prompt, "response": response, "analysis": analysis}
                db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
                display_single_result(response, analysis)

            elif command == "batch-evaluate":
                # Re-implement the core logic of the 'batch-evaluate' command
                session_id = f"playbook_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                console.print(f"Session ID: [bold yellow]{session_id}[/bold yellow]")
                connector = get_connector(params["model"])
                prompts_to_run = prompt_manager.filter_by_category(params["category"])
                
                with Progress() as progress:
                    task = progress.add_task(f"[green]Running batch for '{params['category']}'...", total=len(prompts_to_run))
                    for p in prompts_to_run:
                        response = connector.send_prompt(p)
                        analysis = analyzer.analyze(response, p)
                        result_data = {"prompt": p, "response": response, "analysis": analysis}
                        db_manager.insert_result(convert_result_to_flat_dict(result_data, session_id))
                        progress.update(task, advance=1)
                console.print(f"[bold green]✅ Batch evaluation for '{params['category']}' complete.[/bold green]")
            else:
                console.print(f"[bold red]Skipping step '{step_name}': Unknown command '{command}'.[/bold red]")

        except KeyError as e:
            console.print(f"[bold red]Error in step '{step_name}': Missing required parameter {e}.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred during step '{step_name}': {e}[/bold red]")

    console.rule("[bold green]Playbook execution finished[/bold green]")

if __name__ == "__main__":
    app()
