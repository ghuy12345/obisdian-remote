import pytest

from app.engine import NoteNotFound, WriteRejected
from app.gitrepo import WriteConflict
from app.parser import parse_note
from tests.conftest import NOTES, git


# ---------------------------------------------------------------- parser
def test_parser_links_tags_code():
    n = parse_note("Creative Strategy/Hooks.md", NOTES["Creative Strategy/Hooks.md"])
    targets = [l.target for l in n.links]
    assert targets == ["awareness"]  # code block + inline code ignored
    assert n.links[0].alias == "awareness level"
    assert set(n.tags) == {"copywriting", "hooks"}


def test_parser_sections_and_nested_tags():
    n = parse_note("a.md", NOTES["Creative Strategy/Awareness Levels.md"])
    assert n.aliases == ["Schwartz levels", "awareness"]
    assert "hook/angle" in n.tags and "copywriting" in n.tags
    assert n.headings == ["Awareness Levels", "Unaware", "Most aware"]
    offer = [l for l in n.links if l.target == "Offer Architecture"][0]
    assert offer.section == "Awareness Levels > Most aware"
    assert "offer" in offer.context


def test_parser_broken_frontmatter_does_not_crash():
    n = parse_note("x.md", "---\ntags: [unclosed\n---\nBody [[Link]]\n")
    assert n.errors and [l.target for l in n.links] == ["Link"]


def test_chunking_long_note():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 80 for i in range(20))
    n = parse_note("long.md", "# Title\n\n" + body, chunk_chars=1000, chunk_overlap=100)
    assert len(n.chunks) > 5 and all(len(c.text) <= 1300 for c in n.chunks)


# ---------------------------------------------------------------- graph
def test_link_resolution(setup):
    brain, _, _ = setup
    g = brain.graph
    hooks = "Creative Strategy/Hooks.md"
    aw = "Creative Strategy/Awareness Levels.md"
    assert g.g.has_edge(hooks, aw)                                  # via alias
    assert g.g.has_edge("Offers/Offer Architecture.md", "Creative Strategy/Market Sophistication.md")  # md link
    assert "Unique Mechanism" in g.unresolved                        # unwritten note
    assert "ad-screenshot.png" not in g.unresolved                   # attachment ignored
    assert not any(".obsidian" in p for p in g.notes)


def test_backlinks_with_context(setup):
    brain, _, _ = setup
    bl = brain.backlinks("awareness levels")  # case-insensitive
    srcs = {b["title"] for b in bl["backlinks"]}
    assert srcs == {"Hooks", "Market Sophistication"}
    assert any("awareness level" in c for b in bl["backlinks"] for c in b["contexts"])


def test_path_neighbors_hubs_gaps(setup):
    brain, _, _ = setup
    p = brain.find_path("Men Mansion", "Hooks")
    assert p["connected"] is False
    p = brain.find_path("Offer Architecture", "Unaware" if False else "Market Sophistication")
    assert p["connected"] and p["length"] >= 1
    nb = brain.neighbors("Hooks", hops=2)
    assert {n["title"] for n in nb["neighbors"]} >= {"Awareness Levels", "Market Sophistication", "Offer Architecture"}
    assert brain.hubs(3)[0]["title"] in {"Awareness Levels", "Hooks"}
    gaps = brain.gaps()
    assert gaps["unresolved_links"][0]["target"] == "Unique Mechanism"
    assert set(gaps["orphans"]) == {"Men Mansion", "2026-09-26"}


def test_not_found_suggests(setup):
    brain, _, _ = setup
    with pytest.raises(NoteNotFound) as e:
        brain.resolve("Offer")
    assert "Offer Architecture" in e.value.suggestions


# ---------------------------------------------------------------- search
def test_search_hybrid_and_semantic(setup):
    brain, _, _ = setup
    r = brain.search("Schwartz customer awareness")
    assert r["results"][0]["title"] == "Awareness Levels"
    assert "notice" not in r
    kw = brain.search("tallow", mode="keyword")
    assert [x["title"] for x in kw["results"]] == ["Men Mansion"]
    folder = brain.search("hooks", folder="Offers")
    assert all(x["path"].startswith("Offers/") for x in folder["results"])


def test_related_finds_unlinked(setup):
    brain, _, _ = setup
    rel = brain.related("Men Mansion")
    titles = [r["title"] for r in rel["results"]]
    assert titles and "Men Mansion" not in titles


def test_get_context(setup):
    brain, _, _ = setup
    ctx = brain.get_context("Schwartz levels")  # alias
    assert ctx["note"]["title"] == "Awareness Levels"
    assert {x["title"] for x in ctx["linked_from"]} == {"Hooks", "Market Sophistication"}
    assert "Unique Mechanism" not in ctx["unwritten_links"]
    ctx2 = brain.get_context("opening seconds of video ads")  # free text
    assert ctx2["found"] and ctx2["note"]["title"] in {"Hooks", "Men Mansion"}


# ---------------------------------------------------------------- sync
def test_sync_picks_up_laptop_changes(setup):
    brain, laptop, _ = setup
    laptop.write("Creative Strategy/Unique Mechanism.md", "# Unique Mechanism\nWhy it works. [[Hooks]]\n")
    laptop.delete("Daily/2026-09-26.md")
    stats = brain.sync()
    assert stats["pulled"] and stats["added"] == 1 and stats["deleted"] == 1
    assert "Unique Mechanism" not in brain.graph.unresolved   # link now resolves
    assert brain.graph.g.has_edge("Creative Strategy/Market Sophistication.md",
                                  "Creative Strategy/Unique Mechanism.md")
    assert brain.sync()["pulled"] is False  # idempotent


def test_embeddings_cached(setup):
    brain, laptop, _ = setup
    first = brain.store.counts()["embedded_chunks"]
    laptop.write("Clients/Men Mansion.md", NOTES["Clients/Men Mansion.md"] + "\nNew line about bundles.\n")
    brain.sync()
    # Only the changed note's chunks are re-embedded; old vectors for it are pruned.
    assert brain.store.counts()["embedded_chunks"] == first


# ---------------------------------------------------------------- writes
def test_create_note_roundtrip(setup):
    brain, laptop, _ = setup
    r = brain.create_note("Inbox/Idea", "Test idea linking [[Hooks]]", {"tags": ["inbox"]})
    assert r["created"] == "Inbox/Idea.md"
    assert "Idea" in {b["title"] for b in brain.backlinks("Hooks")["backlinks"]}
    laptop.pull()
    text = laptop.read("Inbox/Idea.md")
    assert text.startswith("---\ntags:\n- inbox\n---") and "[[Hooks]]" in text
    with pytest.raises(WriteRejected):
        brain.create_note("Inbox/Idea.md", "again")


def test_append_preserves_existing(setup):
    brain, laptop, _ = setup
    before = NOTES["Clients/Men Mansion.md"]
    brain.append_to_note("Men Mansion", "Call notes: scale the tallow angle.", heading="2026-09-26")
    laptop.pull()
    after = laptop.read("Clients/Men Mansion.md")
    assert after.startswith(before) and after.endswith("## 2026-09-26\n\nCall notes: scale the tallow angle.\n")


def test_append_after_concurrent_laptop_edit(setup):
    brain, laptop, _ = setup
    laptop.write("Offers/Offer Architecture.md", NOTES["Offers/Offer Architecture.md"] + "Laptop line.\n")
    # Server hasn't synced yet; pull-before-write must pick up the laptop change.
    brain.append_to_note("Offer Architecture", "Server line.")
    laptop.pull()
    text = laptop.read("Offers/Offer Architecture.md")
    assert text.index("Laptop line.") < text.index("Server line.")


def test_write_guards(setup):
    brain, _, _ = setup
    for bad in ("../escape.md", ".obsidian/x.md", "a/../../b.md"):
        with pytest.raises(WriteRejected):
            brain.create_note(bad, "x")
    with pytest.raises(WriteRejected):
        brain.create_note("Inbox/empty.md", "   ")
    brain.s.write_create_dirs = "Inbox"
    with pytest.raises(WriteRejected):
        brain.create_note("Elsewhere/n.md", "x")
    brain.create_note("Inbox/ok.md", "x")


def test_conflict_is_reported_not_merged_badly(setup):
    brain, laptop, remote = setup
    repo = brain.repo
    # Server edits line 1 locally (simulating a commit that raced), laptop pushes a
    # different line 1 first.
    f = brain.s.vault_dir / "Daily/2026-09-26.md"
    f.write_text("Server version.\n")
    laptop.write("Daily/2026-09-26.md", "Laptop version.\n")
    with pytest.raises(WriteConflict):
        repo.commit_and_push(["Daily/2026-09-26.md"], "race")
    assert f.read_text() == "Laptop version.\n"     # laptop wins, clone is clean
    assert git(brain.s.vault_dir, "status", "--porcelain") == ""
    assert brain.sync().get("error") is None


def test_history(setup):
    brain, laptop, _ = setup
    laptop.write("Creative Strategy/Hooks.md", NOTES["Creative Strategy/Hooks.md"] + "More.\n", msg="expand hooks")
    brain.sync()
    h = brain.history("Hooks")
    assert [c["message"] for c in h["commits"]] == ["expand hooks", "initial vault"]


# ---------------------------------------------------------------- edits
AW = "Creative Strategy/Awareness Levels.md"


def test_edit_mid_note(setup):
    brain, laptop, _ = setup
    r = brain.edit_note("Awareness Levels", "Lead with story, not product.",
                        "Lead with story, not product. Problem-first hooks work best.")
    assert r["changed"] and r["chars_removed"] == 0
    laptop.pull()
    text = laptop.read(AW)
    assert "Problem-first hooks work best.\n\n## Most aware" in text
    assert text.startswith(NOTES[AW][:40])


def test_edit_must_match_once(setup):
    brain, _, _ = setup
    with pytest.raises(WriteRejected, match="not found"):
        brain.edit_note("Hooks", "text that isn't there", "x")
    brain.create_note("Inbox/dup.md", "same\nsame\n")
    with pytest.raises(WriteRejected, match="2 times"):
        brain.edit_note("dup", "same", "other")
    brain.edit_note("dup", "same", "other", replace_all=True)
    assert (brain.s.vault_dir / "Inbox/dup.md").read_text() == "other\nother\n"


def test_edit_frontmatter(setup):
    brain, laptop, _ = setup
    brain.edit_note("Hooks", "tags: copywriting, hooks", "tags: copywriting, hooks, video")
    assert "video" in brain.read_note("Hooks")["tags"]


def test_section_replace_append_prepend(setup):
    brain, laptop, _ = setup
    brain.update_section("Awareness Levels", "Unaware", "New unaware body.")
    brain.update_section("Awareness Levels", "## Unaware", "Appended line.", mode="append")
    brain.update_section("Awareness Levels", "unaware", "First line.", mode="prepend")
    laptop.pull()
    text = laptop.read(AW)
    assert "## Unaware\n\nFirst line.\n\nNew unaware body.\n\nAppended line.\n\n## Most aware" in text
    assert "doesn't know they have a problem" not in text
    assert text.endswith("[[Offer Architecture|offer]].\n")  # later section untouched


def test_section_last_and_nested(setup):
    brain, _, _ = setup
    # "Awareness Levels" is the H1: its section runs to the end, including H2s.
    brain.update_section("Awareness Levels", "Most aware", "Just the offer.")
    text = (brain.s.vault_dir / AW).read_text()
    assert text.endswith("## Most aware\n\nJust the offer.\n")
    with pytest.raises(WriteRejected, match="Headings: Awareness Levels, Unaware, Most aware"):
        brain.update_section("Awareness Levels", "Nope", "x")


def test_edit_applies_to_latest_laptop_version(setup):
    brain, laptop, _ = setup
    # Laptop changes a different line; server hasn't synced. The edit must keep it.
    laptop.write(AW, NOTES[AW].replace("Eugene Schwartz", "Eugene M. Schwartz"))
    brain.edit_note("Awareness Levels", "Lead with story", "Open with a story")
    laptop.pull()
    text = laptop.read(AW)
    assert "Eugene M. Schwartz" in text and "Open with a story" in text


def test_edit_stale_text_fails_safely(setup):
    brain, laptop, _ = setup
    laptop.write(AW, NOTES[AW].replace("Lead with story, not product.", "Laptop rewrote this."))
    with pytest.raises(WriteRejected, match="not found"):
        brain.edit_note("Awareness Levels", "Lead with story, not product.", "Server version")
    laptop.pull()
    assert "Laptop rewrote this." in laptop.read(AW)


def test_edit_delete_guard_and_toggle(setup):
    brain, _, _ = setup
    brain.create_note("Inbox/big.md", "x" * 6000)
    with pytest.raises(WriteRejected, match="would remove 6000"):
        brain.edit_note("big", "x" * 6000, "")
    brain.s.edits_enabled = False
    with pytest.raises(WriteRejected, match="disabled"):
        brain.edit_note("Hooks", "Hooks", "H")
