"""Gradio web GUI for Swiss-Prot protein → Sytravon/Genesis library screening.

Launch locally::

    python app.py

Public demo link (while this process is running)::

    python app.py --share
"""

from __future__ import annotations

import argparse
import json
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
from src.uniprot_search import fetch_sequence, search_swissprot

_SCREEN: Optional[ScreenLibrary] = None

_PROTEIN_PICKER_HTML = """
<div class="protein-picker">
  <div class="picker-label">Protein name / gene / accession</div>
  <div class="picker-field">
    <div class="chip" hidden tabindex="0">
      <span class="chip-text"></span>
      <button type="button" class="chip-x" aria-label="Clear selection">&times;</button>
    </div>
    <input
      class="picker-input"
      type="text"
      placeholder="e.g. NPY1R, TSHR, ghrelin receptor…"
      autocomplete="off"
      spellcheck="false"
    />
    <ul class="picker-dropdown" hidden></ul>
  </div>
</div>
"""

_PROTEIN_PICKER_CSS = """
.protein-picker {
    width: 100%;
    font-family: inherit;
}
.picker-label {
    font-size: var(--block-title-text-size, 1rem);
    font-weight: var(--block-title-text-weight, 600);
    margin-bottom: 0.4rem;
    color: var(--body-text-color, inherit);
}
.picker-field {
    position: relative;
    width: 100%;
}
.picker-input {
    box-sizing: border-box;
    width: 100%;
    padding: 0.55rem 0.75rem;
    border: 1px solid var(--border-color-primary, #c0c0c0);
    border-radius: 6px;
    background: var(--input-background-fill, #fff);
    color: inherit;
    font: inherit;
    font-size: 1rem;
    outline: none;
}
.picker-input:focus {
    border-color: var(--color-accent, #2563eb);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--color-accent, #2563eb) 25%, transparent);
}
.chip {
    display: none;
    align-items: center;
    gap: 0.35rem;
    max-width: 100%;
    box-sizing: border-box;
    padding: 0.35rem 0.35rem 0.35rem 0.7rem;
    border: 1px solid var(--border-color-primary, #c0c0c0);
    border-radius: 999px;
    background: var(--background-fill-secondary, #eef2ff);
    color: inherit;
    font: inherit;
    font-size: 0.95rem;
    outline: none;
    cursor: default;
    user-select: text;
}
.chip:not([hidden]) {
    display: inline-flex;
}
.chip[hidden] {
    display: none !important;
}
.chip:focus {
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--color-accent, #2563eb) 30%, transparent);
}
.chip-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: min(100%, 52rem);
}
.chip-x {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem;
    height: 1.5rem;
    border: none;
    border-radius: 999px;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 1.15rem;
    line-height: 1;
    cursor: pointer;
}
.chip-x:hover {
    background: color-mix(in srgb, currentColor 12%, transparent);
}
.picker-dropdown {
    display: none;
    position: absolute;
    top: calc(100% + 2px);
    left: 0;
    right: 0;
    z-index: 50;
    margin: 0;
    padding: 0.25rem 0;
    list-style: none;
    max-height: 16rem;
    overflow-y: auto;
    border: 1px solid var(--border-color-primary, #c0c0c0);
    border-radius: 6px;
    background: var(--background-fill-primary, #fff);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}
.picker-dropdown:not([hidden]) {
    display: block;
}
.picker-dropdown[hidden] {
    display: none !important;
}
.picker-dropdown li {
    padding: 0.55rem 0.75rem;
    cursor: pointer;
    font: inherit;
    font-size: 0.95rem;
}
.picker-dropdown li:hover,
.picker-dropdown li.active {
    background: var(--background-fill-secondary, #f3f4f6);
}
.picker-dropdown .empty {
    color: var(--body-text-color-subdued, #666);
    cursor: default;
}
.picker-dropdown .empty:hover {
    background: transparent;
}
"""

_PROTEIN_PICKER_JS = r"""
(() => {
  let debounceTimer = null;
  let requestId = 0;
  let activeIndex = -1;
  let currentHits = [];

  function qs(sel) {
    return element.querySelector(sel);
  }

  function hideDropdown() {
    const dropdown = qs(".picker-dropdown");
    if (!dropdown) return;
    dropdown.hidden = true;
    dropdown.innerHTML = "";
    activeIndex = -1;
    currentHits = [];
  }

  function showTypingMode() {
    const chip = qs(".chip");
    const chipText = qs(".chip-text");
    const input = qs(".picker-input");
    if (chip) chip.hidden = true;
    if (chipText) chipText.textContent = "";
    if (input) {
      input.hidden = false;
      input.disabled = false;
    }
    hideDropdown();
  }

  function showChipMode(label) {
    const chip = qs(".chip");
    const chipText = qs(".chip-text");
    const input = qs(".picker-input");
    hideDropdown();
    if (input) {
      input.hidden = true;
      input.value = "";
      input.disabled = false;
    }
    if (chipText) chipText.textContent = label || "";
    if (chip) {
      chip.hidden = false;
      chip.focus();
    }
  }

  function setSelection(payload) {
    props.value = payload;
    trigger("change");
  }

  function clearSelection() {
    showTypingMode();
    setSelection(null);
    const input = qs(".picker-input");
    if (input) input.focus();
  }

  function renderDropdown(hits) {
    const dropdown = qs(".picker-dropdown");
    if (!dropdown) return;
    currentHits = Array.isArray(hits) ? hits : [];
    dropdown.innerHTML = "";
    activeIndex = -1;
    if (!currentHits.length) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "No Swiss-Prot matches";
      dropdown.appendChild(li);
      dropdown.hidden = false;
      return;
    }
    currentHits.forEach((hit) => {
      const li = document.createElement("li");
      li.textContent = hit.label || hit.accession || "";
      li.dataset.accession = hit.accession || "";
      dropdown.appendChild(li);
    });
    dropdown.hidden = false;
  }

  function highlightActive() {
    const dropdown = qs(".picker-dropdown");
    if (!dropdown) return;
    const items = dropdown.querySelectorAll("li:not(.empty)");
    items.forEach((item, index) => {
      item.classList.toggle("active", index === activeIndex);
    });
    if (activeIndex >= 0 && items[activeIndex]) {
      items[activeIndex].scrollIntoView({ block: "nearest" });
    }
  }

  async function selectHit(hit) {
    if (!hit || !hit.accession) return;
    const input = qs(".picker-input");
    hideDropdown();
    if (input) input.disabled = true;
    try {
      const resolved = await server.resolve_protein(hit.accession);
      if (!resolved || !resolved.sequence) {
        if (input) input.disabled = false;
        return;
      }
      const label = resolved.label || hit.label || resolved.accession;
      showChipMode(label);
      setSelection({
        accession: resolved.accession,
        sequence: resolved.sequence,
        label: label,
      });
    } catch (err) {
      console.error(err);
      showTypingMode();
      setSelection(null);
    } finally {
      if (input) input.disabled = false;
    }
  }

  async function runSearch(query) {
    const id = ++requestId;
    const q = (query || "").trim();
    if (q.length < 2) {
      hideDropdown();
      return;
    }
    try {
      const hits = await server.search_proteins(q);
      if (id !== requestId) return;
      renderDropdown(hits);
    } catch (err) {
      if (id !== requestId) return;
      const dropdown = qs(".picker-dropdown");
      if (!dropdown) return;
      dropdown.innerHTML = "";
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "Search failed";
      dropdown.appendChild(li);
      dropdown.hidden = false;
      currentHits = [];
    }
  }

  function syncFromValue() {
    const value = props.value;
    if (value && value.label && value.sequence) {
      showChipMode(value.label);
    } else {
      const chip = qs(".chip");
      const input = qs(".picker-input");
      if (chip) chip.hidden = true;
      if (input) input.hidden = false;
      hideDropdown();
    }
  }

  element.addEventListener("input", (event) => {
    if (!event.target.classList.contains("picker-input")) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => runSearch(event.target.value), 300);
  });

  element.addEventListener("keydown", (event) => {
    const input = qs(".picker-input");
    const chip = qs(".chip");
    const dropdown = qs(".picker-dropdown");

    if (chip && !chip.hidden && (event.key === "Backspace" || event.key === "Delete")) {
      if (event.target === chip || chip.contains(event.target) || event.target === element) {
        event.preventDefault();
        clearSelection();
      }
      return;
    }

    if (!input || input.hidden || !dropdown || dropdown.hidden || !currentHits.length) return;
    if (event.target !== input) return;

    if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = Math.min(activeIndex + 1, currentHits.length - 1);
      highlightActive();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      highlightActive();
    } else if (event.key === "Enter") {
      if (activeIndex >= 0 && currentHits[activeIndex]) {
        event.preventDefault();
        selectHit(currentHits[activeIndex]);
      }
    } else if (event.key === "Escape") {
      hideDropdown();
    }
  });

  element.addEventListener("mousedown", (event) => {
    const li = event.target.closest(".picker-dropdown li");
    if (!li || li.classList.contains("empty")) return;
    event.preventDefault();
    const accession = li.dataset.accession;
    const hit = currentHits.find((h) => h.accession === accession) || {
      accession,
      label: li.textContent,
    };
    selectHit(hit);
  });

  element.addEventListener("click", (event) => {
    if (event.target.closest(".chip-x")) {
      event.preventDefault();
      event.stopPropagation();
      clearSelection();
    }
  });

  document.addEventListener("click", (event) => {
    const field = qs(".picker-field");
    if (field && !field.contains(event.target)) {
      hideDropdown();
    }
  });

  watch("value", () => {
    syncFromValue();
  });

  syncFromValue();
})();
"""


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


def search_proteins(query: str) -> list[dict[str, str]]:
    """Search Swiss-Prot and return suggestion dicts for the HTML picker.

    Args:
        query: Free-text protein / gene / accession query.

    Returns:
        A list of ``{"accession", "label"}`` dicts (may be empty).
    """
    try:
        hits = search_swissprot(query, size=10, human_only=True)
    except Exception:  # noqa: BLE001 — surface empty list in the UI dropdown
        return []
    return [{"accession": hit.accession, "label": hit.label} for hit in hits]


def resolve_protein(accession: str) -> dict[str, Any]:
    """Fetch a UniProt sequence and build chip payload fields.

    Args:
        accession: UniProt primary accession.

    Returns:
        Dict with ``accession``, ``label``, ``sequence``, ``gene``,
        ``protein_name``, ``organism``, and ``length``.

    Raises:
        ValueError: If the accession cannot be resolved.
    """
    entry = fetch_sequence(accession)
    gene = entry.gene
    protein_name = entry.protein_name
    if gene and protein_name:
        label = f"{gene} — {protein_name} ({entry.accession})"
    elif protein_name:
        label = f"{protein_name} ({entry.accession})"
    elif gene:
        label = f"{gene} ({entry.accession})"
    else:
        label = entry.accession
    if entry.organism and entry.organism.lower() != "homo sapiens":
        label = f"{label} [{entry.organism}]"
    return {
        "accession": entry.accession,
        "label": label,
        "sequence": entry.sequence,
        "gene": gene,
        "protein_name": protein_name,
        "organism": entry.organism,
        "length": len(entry.sequence),
    }


def on_protein_change(value: Any) -> tuple[str, str]:
    """Parse the HTML picker value into accession and sequence states.

    Args:
        value: Picker payload dict, JSON string, or ``None``.

    Returns:
        Tuple of ``(accession, sequence)``.
    """
    if value is None or value == "" or value == {}:
        return "", ""
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return "", ""
    if not isinstance(payload, dict):
        return "", ""
    accession = str(payload.get("accession") or "").strip()
    sequence = str(payload.get("sequence") or "").strip()
    return accession, sequence


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


_APP_CSS = """
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
#protein-picker {
    margin-bottom: 0.5rem;
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
# GPCR-Predictor Demo

Run AI-based virtual screening on the Sytravon and Genesis ligand libraries.
            """.strip()
        )

        accession_state = gr.State("")
        sequence_state = gr.State("")

        protein_picker = gr.HTML(
            value=None,
            label="Protein",
            html_template=_PROTEIN_PICKER_HTML,
            css_template=_PROTEIN_PICKER_CSS,
            js_on_load=_PROTEIN_PICKER_JS,
            server_functions=[search_proteins, resolve_protein],
            elem_id="protein-picker",
            padding=False,
            container=False,
        )

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

        protein_picker.change(
            fn=on_protein_change,
            inputs=[protein_picker],
            outputs=[accession_state, sequence_state],
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
        css=_APP_CSS,
    )


if __name__ == "__main__":
    main()
