"""Parse an Obsidian markdown note into structured data.

Pure functions, no I/O besides reading the text you pass in.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import unquote

import frontmatter
import yaml

FENCE_RE = re.compile(r"^(```|~~~)")
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
WIKILINK_RE = re.compile(r"(!?)\[\[([^\[\]\n]+?)\]\]")
MDLINK_RE = re.compile(r"(!?)\[([^\]\n]*)\]\(([^)\s]+?\.md)(#[^)\s]*)?\)")
TAG_RE = re.compile(r"(?:(?<=\s)|^)#([A-Za-z0-9_\-/À-￿]*[A-Za-z_\-/À-￿][A-Za-z0-9_\-/À-￿]*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


@dataclass
class Link:
    target: str          # raw target as written, without heading/block/alias
    kind: str            # "wikilink" | "embed" | "markdown"
    heading: str | None  # "#Heading" or "^block" part, if any
    alias: str | None
    context: str         # the line the link sits on (trimmed)
    section: str         # heading path the link sits under


@dataclass
class Chunk:
    ord: int
    heading: str
    text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class ParsedNote:
    path: str
    title: str
    frontmatter: dict
    body: str
    tags: list[str]
    aliases: list[str]
    links: list[Link]
    chunks: list[Chunk]
    headings: list[str]
    word_count: int
    content_hash: str
    errors: list[str] = field(default_factory=list)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value)]


def _json_safe(obj):
    """Frontmatter can contain dates etc. Make it JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _split_frontmatter(text: str) -> tuple[dict, str, list[str]]:
    try:
        post = frontmatter.loads(text)
        return _json_safe(dict(post.metadata)), post.content, []
    except (yaml.YAMLError, ValueError) as e:  # broken YAML should never kill indexing
        body = text
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                body = text[end + 4 :].lstrip("\n")
        return {}, body, [f"frontmatter: {e.__class__.__name__}"]


def _code_mask(lines: list[str]) -> list[bool]:
    """True for lines inside fenced code blocks."""
    mask, inside = [], False
    for line in lines:
        if FENCE_RE.match(line.strip()):
            mask.append(True)
            inside = not inside
            continue
        mask.append(inside)
    return mask


def _parse_link_inner(inner: str) -> tuple[str, str | None, str | None]:
    alias = None
    if "|" in inner:
        inner, alias = inner.split("|", 1)
        alias = alias.strip() or None
    heading = None
    for sep in ("#", "^"):
        if sep in inner:
            idx = inner.index(sep)
            heading = inner[idx:]
            inner = inner[:idx]
            break
    return inner.strip(), heading, alias


def chunk_sections(sections: list[tuple[str, str]], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in sections:
        text = text.strip()
        if not text:
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buf = ""
        for p in paras:
            while len(p) > size:  # very long paragraph: hard split
                head, p = p[:size], p[size - overlap :]
                if buf:
                    chunks.append(Chunk(len(chunks), heading, buf))
                    buf = ""
                chunks.append(Chunk(len(chunks), heading, head))
            if buf and len(buf) + len(p) + 2 > size:
                chunks.append(Chunk(len(chunks), heading, buf))
                buf = buf[-overlap:] + "\n\n" + p if overlap else p
            else:
                buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append(Chunk(len(chunks), heading, buf))
    return chunks


def parse_note(path: str, text: str, chunk_chars: int = 1500, chunk_overlap: int = 200) -> ParsedNote:
    fm, body, errors = _split_frontmatter(text)
    title = PurePosixPath(path).stem

    lines = body.split("\n")
    in_code = _code_mask(lines)

    links: list[Link] = []
    tags: set[str] = {t.lstrip("#") for t in _as_list(fm.get("tags") or fm.get("tag"))}
    headings: list[str] = []
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [("", [])]

    for line, code in zip(lines, in_code):
        if code:
            sections[-1][1].append(line)
            continue
        m = HEADING_RE.match(line)
        if m:
            level, name = len(m.group(1)), m.group(2).strip()
            headings.append(name)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, name))
            sections.append((" > ".join(h for _, h in heading_stack), [line]))
        else:
            sections[-1][1].append(line)

        section = sections[-1][0]
        scan = INLINE_CODE_RE.sub("", line)
        context = scan.strip()[:300]
        for bang, inner in WIKILINK_RE.findall(scan):
            target, heading, alias = _parse_link_inner(inner)
            if not target:  # [[#Heading]] = link to self
                continue
            links.append(Link(target, "embed" if bang else "wikilink", heading, alias, context, section))
        for bang, label, target, heading in MDLINK_RE.findall(scan):
            if "://" in target:
                continue
            links.append(Link(unquote(target), "markdown", heading or None, label or None, context, section))
        if not m:
            for tag in TAG_RE.findall(scan):
                tags.add(tag.rstrip("/"))

    chunks = chunk_sections([(h, "\n".join(ls)) for h, ls in sections], chunk_chars, chunk_overlap)

    return ParsedNote(
        path=path,
        title=title,
        frontmatter=fm,
        body=body,
        tags=sorted(tags),
        aliases=_as_list(fm.get("aliases") or fm.get("alias")),
        links=links,
        chunks=chunks,
        headings=headings,
        word_count=len(body.split()),
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        errors=errors,
    )
