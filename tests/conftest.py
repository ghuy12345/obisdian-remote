import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app import engine as engine_mod
from app.config import Settings

NOTES = {
    "Creative Strategy/Awareness Levels.md": """---
aliases: [Schwartz levels, awareness]
tags: [copywriting]
---
# Awareness Levels

Eugene Schwartz defined five stages of customer awareness. See [[Market Sophistication]]
and [[Hooks|hook writing]]. Each level needs a different #hook/angle.

## Unaware
The prospect doesn't know they have a problem. Lead with story, not product.

## Most aware
They know the product, just give them the [[Offer Architecture|offer]].
""",
    "Creative Strategy/Market Sophistication.md": """# Market Sophistication

How many claims the market has already heard. Tied to [[Awareness Levels]].
Stage three markets need a new mechanism, see [[Unique Mechanism]].
""",
    "Creative Strategy/Hooks.md": """---
tags: copywriting, hooks
---
# Hooks

The first three seconds of a video ad decide whether anyone watches.
Match the hook to the [[awareness|awareness level]] of the viewer.

```python
# [[Not A Link]] inside code
```

Inline `[[also not a link]]` either.
""",
    "Offers/Offer Architecture.md": """# Offer Architecture

Bundles, guarantees and price anchoring. Strong offers make [[Hooks]] easier to write.
Linked from a markdown link: [mechanism](../Creative%20Strategy/Market%20Sophistication.md)
""",
    "Clients/Men Mansion.md": """# Men Mansion

Client. Tallow skincare for men. Video ads with strong opening seconds perform best.
Screenshot: ![[ad-screenshot.png]]
""",
    "Daily/2026-09-26.md": "Nothing linked here.\n",
}


def git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


class Laptop:
    """A second clone standing in for the Obsidian laptop."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, rel, text, msg="laptop edit"):
        f = self.path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        git(self.path, "add", "-A")
        git(self.path, "commit", "-m", msg)
        git(self.path, "push", "origin", "HEAD:main")

    def delete(self, rel):
        git(self.path, "rm", rel)
        git(self.path, "commit", "-m", "delete")
        git(self.path, "push", "origin", "HEAD:main")

    def pull(self):
        git(self.path, "pull", "--rebase", "origin", "main")

    def read(self, rel):
        return (self.path / rel).read_text()


class FakeEmbedder:
    """Deterministic bag-of-words vectors so semantic tests are meaningful and offline."""

    def __init__(self, *a, **k):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            out.append(v)
        return out


@pytest.fixture
def setup(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    laptop_dir = tmp_path / "laptop"
    git(tmp_path, "clone", str(remote), str(laptop_dir))
    git(laptop_dir, "config", "user.name", "Josh")
    git(laptop_dir, "config", "user.email", "josh@example.com")
    git(laptop_dir, "checkout", "-b", "main")
    (laptop_dir / ".obsidian").mkdir()
    (laptop_dir / ".obsidian" / "app.json").write_text("{}")
    for rel, text in NOTES.items():
        f = laptop_dir / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    git(laptop_dir, "add", "-A")
    git(laptop_dir, "commit", "-m", "initial vault")
    git(laptop_dir, "push", "-u", "origin", "main")

    monkeypatch.setattr(engine_mod, "Embedder", FakeEmbedder)
    settings = Settings(
        vault_repo_url=f"file://{remote}", data_dir=tmp_path / "data", embeddings_api_key="fake",
        api_token="t", mcp_secret="s", _env_file=None,
    )
    brain = engine_mod.Brain(settings)
    brain.startup()
    return brain, Laptop(laptop_dir), remote
