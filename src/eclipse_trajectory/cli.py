from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from eclipse_trajectory.config import load_config
from eclipse_trajectory.export.timeline import write_timeline
from eclipse_trajectory.ingest.video import inspect_video
from eclipse_trajectory.pipeline import deterministic_actions, run_local_inference, run_pipeline
from eclipse_trajectory.schemas import ActionRecord, CandidateEvent
from eclipse_trajectory.synthesis.hierarchy import synthesize_hierarchy
from eclipse_trajectory.util import atomic_write_jsonl, read_jsonl

app = typer.Typer(
    name="eclipse-trajectory",
    help="Local-first evidence extraction from Eclipse screen recordings.",
    no_args_is_help=True,
)


@app.command("inspect")
def inspect_command(
    input_video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Inspect technical metadata and calculate the stable source hash."""
    metadata = inspect_video(input_video)
    typer.echo(json.dumps(metadata.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command("run")
def run_command(
    input_video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/offline.yaml"),
) -> None:
    """Run inspection, detection, evidence extraction, fallback inference, and export."""
    session_dir = run_pipeline(input_video, load_config(config))
    typer.echo(str(session_dir))


@app.command("preprocess")
def preprocess_command(
    input_video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/offline.yaml"),
) -> None:
    """Create the complete deterministic extraction baseline (resumable by stage)."""
    session_dir = run_pipeline(input_video, load_config(config))
    typer.echo(str(session_dir))


@app.command("detect")
def detect_command(
    session_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Validate and report the already extracted window/event records."""
    windows = list(read_jsonl(session_directory / "windows.jsonl"))
    events = list(read_jsonl(session_directory / "candidate_events.jsonl"))
    typer.echo(json.dumps({"windows": len(windows), "candidate_events": len(events)}))


@app.command("infer-actions")
def infer_actions_command(
    session_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    backend: Annotated[str, typer.Option("--backend")] = "deterministic",
    endpoint: Annotated[str, typer.Option("--endpoint")] = "http://127.0.0.1:8000/v1",
    model: Annotated[str | None, typer.Option("--model")] = None,
    redact_likely_identifiers: Annotated[
        bool, typer.Option("--redact-likely-identifiers/--retain-identifiers")
    ] = True,
) -> None:
    """Infer actions with the deterministic fallback or a localhost multimodal server."""
    if backend == "deterministic":
        events = [
            CandidateEvent.model_validate(item)
            for item in read_jsonl(session_directory / "candidate_events.jsonl")
        ]
        actions = deterministic_actions(events)
        atomic_write_jsonl(session_directory / "actions.jsonl", actions)
        synthesize_hierarchy(session_directory)
        write_timeline(session_directory, actions)
    elif backend == "local-openai-compatible":
        if not model:
            raise typer.BadParameter("--model is required for local-openai-compatible")
        actions = run_local_inference(
            session_directory,
            endpoint,
            model,
            redact_likely_identifiers=redact_likely_identifiers,
        )
    else:
        raise typer.BadParameter("--backend must be deterministic or local-openai-compatible")
    typer.echo(json.dumps({"actions": len(actions), "backend": backend}))


@app.command("synthesize")
def synthesize_command(
    session_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Reconcile ordered action records into episodes and a session summary."""
    episodes, session = synthesize_hierarchy(session_directory)
    typer.echo(
        json.dumps(
            {
                "episodes": len(episodes),
                "executive_summary_available": bool(session["executive_summary"]),
            }
        )
    )


@app.command("review")
def review_command(
    session_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Open the local evidence timeline (correction editing is not yet implemented)."""
    timeline = (session_directory / "timeline.html").resolve()
    if not timeline.exists():
        actions = [
            ActionRecord.model_validate(item)
            for item in read_jsonl(session_directory / "actions.jsonl")
        ]
        write_timeline(session_directory, actions)
    if open_browser:
        webbrowser.open(timeline.as_uri())
    typer.echo(str(timeline))


@app.command("export")
def export_command(
    session_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_format: Annotated[str, typer.Option("--format")] = "jsonl",
) -> None:
    """Validate the baseline local export and print its artifact paths."""
    if output_format != "jsonl":
        raise typer.BadParameter("The first release supports only --format jsonl")
    required = ["actions.jsonl", "episodes.jsonl", "session.json", "manifest.json"]
    missing = [name for name in required if not (session_directory / name).exists()]
    if missing:
        raise typer.BadParameter(f"Missing export artifacts: {', '.join(missing)}")
    typer.echo(json.dumps({name: str((session_directory / name).resolve()) for name in required}))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
