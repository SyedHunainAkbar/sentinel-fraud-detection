"""Load and chunk the policy corpus for retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config


@dataclass
class Chunk:
    id: str
    source: str
    heading: str
    text: str


def load_chunks(policies_dir: Path | None = None) -> list[Chunk]:
    """Split each policy markdown file into section chunks with stable ids."""
    policies_dir = policies_dir or (config.DATA_DIR / "policies")
    chunks: list[Chunk] = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        heading = path.stem
        buf: list[str] = []
        for line in path.read_text().splitlines():
            if line.startswith("## "):
                if buf:
                    _flush(chunks, path.name, heading, buf)
                    buf = []
                heading = line[3:].strip()
            else:
                buf.append(line)
        if buf:
            _flush(chunks, path.name, heading, buf)
    return chunks


def _flush(chunks, source, heading, buf):
    text = "\n".join(buf).strip()
    if text:
        chunks.append(Chunk(id=f"{source}#{len(chunks)}", source=source,
                            heading=heading, text=text))
