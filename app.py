"""Gradio web GUI for Swiss-Prot protein → Sytravon/Genesis library screening.

Launch locally::

    python app.py

Public demo link (while this process is running)::

    python app.py --share
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any, Optional

import gradio as gr
from PIL import Image

from src.screen_library import (
    DEFAULT_LIGAND_MAP,
    DEFAULT_MODEL,
    ScreenLibrary,
)
from src.uniprot_search import UniProtHit, fetch_sequence, search_swissprot

_SCREEN: Optional[ScreenLibrary] = None
_LABEL_TO_HIT: dict[str, UniProtHit] = {}


def _get_screen(model_path: str, ligand_map_path: str) -> ScreenLibrary:
    """Return the process-wide :class:`ScreenLibrary`, creating it on first use.

    Args:
        model_path: Joblib model path.
        ligand_map_path: Ligand ID map TSV path.

    Returns:
        A warmed :class:`ScreenLibrary` instance.
    """
    global _SCREEN
    if _SCREEN is None:
        _SCREEN = ScreenLibrary(
            model_path=model_path,
            ligand_map_path=ligand_map_path,
            verbose=True,
        )
        _SCREEN.warm_ligands()
    return _SCREEN


def on_search(query: str) -> Any:
    """Update Swiss-Prot suggestions under the search box.

    Args:
        query: Free-text protein / gene / accession query.

    Returns:
        A Gradio Radio update with clickable suggestion labels.
    """
    global _LABEL_TO_HIT
    try:
        hits = search_swissprot(query, size=10, human_only=True)
    except Exception as exc:  # noqa: BLE001 — surface API errors in the UI
        _LABEL_TO_HIT = {}
        return gr.update(
            choices=[],
            value=None,
            label=f"Suggestions (search failed: {exc})",
        )
    _LABEL_TO_HIT = {hit.label: hit for hit in hits}
    labels = [hit.label for hit in hits]
    return gr.update(
        choices=labels,
        value=None,
        label="Suggestions (click to select)",
    )


def on_select_suggestion(
    label: Optional[str],
) -> tuple[str, str, str]:
    """Resolve a clicked suggestion to a UniProt sequence.

    Args:
        label: Selected Radio label.

    Returns:
        Tuple of ``(status_markdown, accession, sequence)``.
    """
    if not label or label not in _LABEL_TO_HIT:
        return (
            "_No protein selected._",
            "",
            "",
        )
    hit = _LABEL_TO_HIT[label]
    try:
        entry = fetch_sequence(hit.accession)
    except Exception as exc:  # noqa: BLE001
        return (
            f"**Failed to fetch sequence for {hit.accession}:** {exc}",
            "",
            "",
        )
    name = entry.gene or entry.protein_name or entry.accession
    status = (
        f"**Selected:** {name}  \n"
        f"**Accession:** `{entry.accession}`  \n"
        f"**Organism:** {entry.organism or 'n/a'}  \n"
        f"**Sequence length:** {len(entry.sequence)} aa"
    )
    return status, entry.accession, entry.sequence


def _format_bytes(n_bytes: int) -> str:
    """Format a byte count for a download button label.

    Args:
        n_bytes: Size in bytes.

    Returns:
        A short human-readable size such as ``14.6 MB``.
    """
    mb = n_bytes / (1024 * 1024)
    if mb >= 10:
        return f"{mb:.0f} MB"
    if mb >= 1:
        return f"{mb:.1f} MB"
    kb = n_bytes / 1024
    if kb >= 1:
        return f"{kb:.0f} KB"
    return f"{n_bytes} B"


def on_run_screen(
    sequence: str,
    accession: str,
    model_path: str,
    ligand_map_path: str,
) -> tuple[list[tuple[Any, str]], Any, str]:
    """Screen the selected protein against Sytravon + Genesis.

    Args:
        sequence: Amino-acid sequence.
        accession: UniProt accession (for CSV naming).
        model_path: Joblib model path.
        ligand_map_path: Ligand ID map path.

    Returns:
        Tuple of ``(gallery_items, download_button_update, status)``.
    """
    seq = str(sequence or "").strip()
    if not seq:
        return (
            [],
            gr.update(value=None, label="Download Full Results", interactive=False),
            "No protein selected.",
        )

    try:
        screen = _get_screen(model_path, ligand_map_path)
        results = screen.screen(seq)
        hits = screen.top_hits(results, k=10)
    except Exception as exc:  # noqa: BLE001
        return (
            [],
            gr.update(value=None, label="Download Full Results", interactive=False),
            f"Error: {exc}",
        )

    gallery: list[tuple[Any, str]] = []
    for hit in hits:
        caption = hit.smiles
        if hit.image is not None:
            gallery.append((hit.image, caption))
        else:
            blank = Image.new("RGB", (320, 240), color=(255, 255, 255))
            gallery.append((blank, caption))

    csv_bytes = ScreenLibrary.results_to_csv_bytes(results)
    acc = str(accession or "protein").strip().upper() or "protein"
    fd, csv_path = tempfile.mkstemp(prefix=f"screen_{acc}_", suffix=".csv")
    os.close(fd)
    with open(csv_path, "wb") as handle:
        handle.write(csv_bytes)

    size_label = _format_bytes(len(csv_bytes))
    download_update = gr.update(
        value=csv_path,
        label=f"Download Full Results ({size_label})",
        interactive=True,
    )
    status = f"Scored {len(results):,} ligands."
    return gallery, download_update, status


_GALLERY_CAPTION_CSS = """
html, body, .gradio-container,
.gradio-container button, .gradio-container input,
.gradio-container textarea, .gradio-container label,
.gradio-container .prose, .gradio-container markdown {
    font-family: "Sinhala Sangam MN", Georgia, serif !important;
}
.caption {
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-all !important;
    text-overflow: clip !important;
    overflow-y: auto !important;
    max-height: 28vh;
    max-width: 100% !important;
    align-self: stretch !important;
}
/* Hide the preview strip of all gallery thumbnails (covers SMILES). */
.preview .thumbnails {
    display: none !important;
}
/* Grid tile captions: span bottom edge; ellipsis inside the tile. */
.caption-label {
    left: var(--block-label-margin) !important;
    right: var(--block-label-margin) !important;
    max-width: none !important;
    width: auto !important;
    text-align: left !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    text-overflow: ellipsis !important;
}
/* Full-width prominent CSV download button. */
#download-full-results button {
    width: 100% !important;
    min-height: 3.25rem !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
}
"""


def build_app(model_path: str, ligand_map_path: str) -> gr.Blocks:
    """Construct the Gradio Blocks UI.

    Args:
        model_path: Joblib model path (from CLI ``--model``).
        ligand_map_path: Ligand map path (from CLI ``--ligand-map``).

    Returns:
        A Gradio ``Blocks`` app.
    """
    with gr.Blocks(title="GPCR Library Screen") as demo:
        gr.Markdown(
            """
# GPCR sequence library screen

Search **Swiss-Prot** (human) for a protein, then score it against the
**Sytravon** + **Genesis** ligand libraries. Top 10 hits are shown with
structures; download the full CSV of `SMILES`, `ID`, `dataset`, and `P(Active)`.
            """.strip()
        )

        accession_state = gr.State("")
        sequence_state = gr.State("")

        with gr.Row():
            query = gr.Textbox(
                label="Protein name / gene / accession",
                placeholder="e.g. NPY1R, TSHR, ghrelin receptor…",
                scale=3,
            )
            search_btn = gr.Button("Search", scale=1)
        suggestions = gr.Radio(
            choices=[],
            label="Suggestions (click to select)",
            interactive=True,
        )
        selected = gr.Markdown("_No protein selected._")

        run_btn = gr.Button("Run library screen", variant="primary")
        status = gr.Markdown("")

        gallery = gr.Gallery(
            label="Top 10 hits",
            columns=2,
            height="auto",
            object_fit="contain",
        )
        download_btn = gr.DownloadButton(
            label="Download Full Results",
            value=None,
            variant="primary",
            size="lg",
            interactive=False,
            elem_id="download-full-results",
        )

        def _run(sequence: str, accession: str) -> tuple[list[tuple[Any, str]], Any, str]:
            """Run screening with the app's fixed model and ligand map paths.

            Args:
                sequence: Amino-acid sequence.
                accession: UniProt accession (for CSV naming).

            Returns:
                Tuple of ``(gallery_items, download_button_update, status)``.
            """
            return on_run_screen(sequence, accession, model_path, ligand_map_path)

        search_btn.click(fn=on_search, inputs=[query], outputs=[suggestions])
        query.submit(fn=on_search, inputs=[query], outputs=[suggestions])
        suggestions.change(
            fn=on_select_suggestion,
            inputs=[suggestions],
            outputs=[selected, accession_state, sequence_state],
        )
        run_btn.click(
            fn=_run,
            inputs=[sequence_state, accession_state],
            outputs=[gallery, download_btn, status],
        )

    return demo


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the screening GUI.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="GPCR sequence library screening GUI")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Joblib model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--ligand-map",
        default=DEFAULT_LIGAND_MAP,
        help=f"Ligand ID map TSV (default: {DEFAULT_LIGAND_MAP})",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio share URL (*.gradio.live).",
    )
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Bind port (default: 7860).",
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="Skip ligand feature warm-up at startup (warms on first Run).",
    )
    args = parser.parse_args(argv)

    if not args.no_warm:
        print("[app] loading model and warming ligand library…", flush=True)
        _get_screen(args.model, args.ligand_map)

    demo = build_app(args.model, args.ligand_map)
    demo.queue().launch(
        share=bool(args.share),
        server_name=args.server_name,
        server_port=int(args.server_port),
        css=_GALLERY_CAPTION_CSS,
    )


if __name__ == "__main__":
    main()
