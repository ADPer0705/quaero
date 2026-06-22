#!/usr/bin/env python3
"""
Quaero CLI - High-Performance Local RAG Engine
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from quaero.core import (
    ensure_directories,
    QuaeroDatabase,
    ingest_pipeline,
    query_rag,
)
from quaero.core.utils import read_config, write_config, get_app_version
from quaero.core.config import QUAERO_CONFIG_PATH, CONFIGS

# ==================================================
# Typer App Initialization
# ==================================================
app = typer.Typer(
    help="Quaero: AI-Powered Local Intelligence & Research Assistant",
    no_args_is_help=True,
    add_completion=False
)
console = Console()

# Sub-apps for clean namespaces
config_app = typer.Typer(help="Manage system configurations and AI provider settings.", no_args_is_help=True)
db_app = typer.Typer(help="Manage the LanceDB vector store directly.", no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(db_app, name="db")

# ==================================================
# Helpers
# ==================================================
def _cast_config_value(value: str):
    """Smart caster for TOML string inputs."""
    if value.lower() in ['true', 'false']: return value.lower() == 'true'
    if value.lstrip('-').isdigit(): return int(value)
    try: return float(value)
    except ValueError: pass
    return value

@app.callback()
def system_check():
    """Executes before any command to ensure the system environment is stable."""
    try:
        ensure_directories()
    except Exception as e:
        console.print(f"[bold red]System Error:[/bold red] Failed to initialize directories: {e}")
        raise typer.Exit(code=1)

# ==================================================
# Top-Level Commands
# ==================================================

@app.command()
def version():
    """Displays the current version of the Quaero CLI."""
    console.print(f"[bold cyan]Quaero CLI Engine[/bold cyan] v{get_app_version()}")

@app.command()
def setup():
    """
    Interactive onboarding wizard to configure core AI engines.
    """
    console.print(Panel.fit("[bold cyan]Quaero Initial Setup[/bold cyan]\nLet's configure your local intelligence engines.", border_style="cyan"))
    
    current_configs = read_config(QUAERO_CONFIG_PATH) or {}
    
    inf_model = Prompt.ask("Primary Inference Model", default=CONFIGS["INFERENCE_MODEL"])
    emb_model = Prompt.ask("Primary Embedding Model", default=CONFIGS["EMBEDDING_MODEL"])
    chunk_size = Prompt.ask("Document Chunk Size (Characters)", default=str(CONFIGS["CHUNK_SIZE"]))
    
    current_configs["INFERENCE_MODEL"] = inf_model
    current_configs["EMBEDDING_MODEL"] = emb_model
    current_configs["CHUNK_SIZE"] = int(chunk_size)
    
    write_config(QUAERO_CONFIG_PATH, current_configs)
    console.print("\n✅ [bold green]Configuration saved![/bold green] You are ready to start ingesting data.")

@app.command()
def status():
    """Displays the real-time status of the local vector database."""
    db = QuaeroDatabase()
    try:
        files_table = db.db.open_table("files")
        chunks_table = db.db.open_table("chunks")
        file_count = files_table.count_rows()
        chunk_count = chunks_table.count_rows()
    except Exception:
        file_count, chunk_count = 0, 0

    table = Table(title="Quaero Engine Status", box=None, title_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    table.add_row("Database State", "[bold green]ONLINE[/bold green]" if chunk_count > 0 else "[yellow]EMPTY[/yellow]")
    table.add_row("Indexed Files", str(file_count))
    table.add_row("Vector Chunks", str(chunk_count))

    console.print(Panel(table, border_style="blue", expand=False))

@app.command()
def ingest(target: Path = typer.Argument(..., help="Path to a file or directory to index")):
    """
    Processes and ingests documents into the vector engine.
    """
    if not target.exists():
        console.print(f"[bold red]Error:[/bold red] Path '{target}' does not exist.")
        raise typer.Exit(code=1)

    db = QuaeroDatabase()
    files_to_process = [target] if target.is_file() else list(target.rglob("*"))
    # Filter for supported extensions to keep the pipeline clean
    valid_exts = {'.pdf', '.txt', '.md', '.csv', '.docx', '.pptx', '.rs', '.py', '.nix', '.toml'}
    files_to_process = [f for f in files_to_process if f.is_file() and f.suffix.lower() in valid_exts]

    if not files_to_process:
        console.print("[yellow]No supported files found to index.[/yellow]")
        raise typer.Exit()

    processed, skipped = 0, 0
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Indexing {len(files_to_process)} files...", total=len(files_to_process))
        for file_path in files_to_process:
            progress.update(task, description=f"Processing: [cyan]{file_path.name}[/cyan]")
            try:
                if ingest_pipeline(file_path, db): processed += 1
                else: skipped += 1
            except Exception as e:
                console.print(f"\n[red]Failed on {file_path.name}:[/red] {e}")
            progress.advance(task)

    console.print(Panel(f"✅ [bold green]Ingestion Complete[/bold green]\nNew/Updated: {processed}\nSkipped (Unchanged): {skipped}", border_style="green", expand=False))

@app.command()
def sync():
    """
    Reconciles the database with the filesystem. 
    Removes orphaned vectors and updates modified files.
    """
    db = QuaeroDatabase()
    try:
        table = db.db.open_table("files")
        # Extract tracked file records safely to native Python lists
        tracked_files = table.to_arrow().to_pylist()
    except Exception:
        console.print("[yellow]Database is empty. Nothing to sync.[/yellow]")
        raise typer.Exit()

    purged, updated, untouched = 0, 0, 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Synchronizing system state...", total=len(tracked_files))
        
        for record in tracked_files:
            file_path = Path(record["absolute_path"])
            file_id = record["file_id"]
            
            # 1. Prune missing files
            if not file_path.exists():
                progress.update(task, description=f"Purging orphan: [red]{file_path.name}[/red]")
                db.remove_file(file_id)
                purged += 1
            else:
                # 2. Update modified files using the native pipeline logic
                progress.update(task, description=f"Checking: [cyan]{file_path.name}[/cyan]")
                try:
                    if ingest_pipeline(file_path, db): updated += 1
                    else: untouched += 1
                except Exception as e:
                    console.print(f"\n[red]Sync failed on {file_path.name}:[/red] {e}")
                    
            progress.advance(task)

    console.print(Panel(
        f"🔄 [bold cyan]System Synchronization Complete[/bold cyan]\n"
        f"Orphaned Files Purged: [red]{purged}[/red]\n"
        f"Modified Files Updated: [green]{updated}[/green]\n"
        f"Pristine Files Skipped: [dim]{untouched}[/dim]",
        border_style="cyan", expand=False
    ))

@app.command()
def chat(query: str = typer.Argument(None, help="A single question to ask immediately")):
    """Launch the interactive RAG interface."""
    console.print(Panel.fit("[bold cyan]Quaero Terminal[/bold cyan]\n[dim]Type your question below. Type 'exit' or 'quit' to close.[/dim]", border_style="cyan"))

    if query:
        query_rag(query)
        return

    while True:
        try:
            user_input = console.input("\n[bold green]❯[/bold green] ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Shutting down systems...[/dim]")
                break
            query_rag(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]Session terminated.[/dim]")
            break

# ==================================================
# Sub-App: Config
# ==================================================
@config_app.command("show")
def config_show():
    """Displays the current active configuration settings."""
    table = Table(title="Quaero Configuration", box=None, title_style="bold magenta")
    table.add_column("Key", style="cyan", justify="right")
    table.add_column("Value", style="green")

    for key in sorted(CONFIGS.keys()):
        display_val = str(CONFIGS[key])
        if len(display_val) > 60: display_val = display_val[:57] + "..."
        table.add_row(key, display_val)

    console.print(Panel(table, border_style="magenta", expand=False))

@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)):
    """Updates a specific configuration setting."""
    normalized_key = key.upper()
    if normalized_key not in CONFIGS:
        console.print(f"[bold red]Error:[/bold red] Unknown configuration key '{normalized_key}'.")
        raise typer.Exit(code=1)

    typed_value = _cast_config_value(value)
    current_configs = read_config(QUAERO_CONFIG_PATH) or {}
    current_configs[normalized_key] = typed_value
    
    write_config(QUAERO_CONFIG_PATH, current_configs)
    console.print(f"✅ [bold green]Success:[/bold green] [cyan]{normalized_key}[/cyan] updated to [yellow]{typed_value}[/yellow]")

# ==================================================
# Sub-App: Database (db)
# ==================================================
@db_app.command("reset")
def db_reset():
    """Destroys the entire vector index and tracked file history. Unrecoverable."""
    console.print("[bold red]WARNING: This will permanently delete all vectors and tracked history.[/bold red]")
    if Confirm.ask("Are you absolutely sure you want to nuke the database?"):
        db = QuaeroDatabase()
        try:
            db.db.drop_table("files")
            db.db.drop_table("chunks")
            console.print("✅ [bold green]Database successfully wiped.[/bold green] System is completely clean.")
        except Exception as e:
            console.print(f"[bold red]Error dropping tables:[/bold red] {e}")

if __name__ == "__main__":
    app()