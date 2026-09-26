"""The link graph: Obsidian-style link resolution plus graph queries."""
from __future__ import annotations

import posixpath
from collections import defaultdict

import networkx as nx

ATTACHMENT_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf", ".mp3", ".mp4", ".mov",
    ".webm", ".wav", ".m4a", ".canvas", ".excalidraw", ".base", ".csv", ".zip",
}


def _norm(s: str) -> str:
    return s.strip().lower()


class Resolver:
    """Resolve a link target the way Obsidian does: by path, then by file name, then by alias."""

    def __init__(self, notes: list[dict]):
        self.by_path: dict[str, str] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.by_alias: dict[str, list[str]] = defaultdict(list)
        for n in notes:
            p = n["path"]
            self.by_path[_norm(p[:-3] if p.endswith(".md") else p)] = p
            self.by_name[_norm(n["title"])].append(p)
            for a in n["aliases"]:
                self.by_alias[_norm(a)].append(p)
        for d in (self.by_name, self.by_alias):
            for k in d:
                d[k].sort(key=lambda x: (x.count("/"), len(x)))  # prefer shallowest

    def resolve(self, target: str, src: str | None = None, kind: str = "wikilink") -> str | None:
        t = target.strip()
        ext = posixpath.splitext(t)[1].lower()
        if ext in ATTACHMENT_EXTS:
            return None
        if t.lower().endswith(".md"):
            t = t[:-3]
        if kind == "markdown" and src:
            rel = posixpath.normpath(posixpath.join(posixpath.dirname(src), t))
            if _norm(rel) in self.by_path:
                return self.by_path[_norm(rel)]
        t = t.lstrip("./").lstrip("/")
        key = _norm(t)
        if key in self.by_path:
            return self.by_path[key]
        if "/" in key:
            for p_key, p in self.by_path.items():
                if p_key.endswith("/" + key):
                    return p
            key = key.rsplit("/", 1)[1]
        if key in self.by_name:
            return self.by_name[key][0]
        if key in self.by_alias:
            return self.by_alias[key][0]
        return None

    def find(self, name: str) -> list[str]:
        """Candidate notes for a user-supplied name/path (for tools)."""
        key = _norm(name[:-3] if name.lower().endswith(".md") else name)
        if key in self.by_path:
            return [self.by_path[key]]
        hits = list(self.by_name.get(key, [])) + [p for p in self.by_alias.get(key, []) if p not in self.by_name.get(key, [])]
        if hits:
            return hits
        return [p for k, p in self.by_path.items() if k.endswith("/" + key)]


class VaultGraph:
    def __init__(self, notes: list[dict], links: list[dict]):
        self.resolver = Resolver(notes)
        self.notes = {n["path"]: n for n in notes}
        self.g = nx.DiGraph()
        self.unresolved: dict[str, list[str]] = defaultdict(list)  # target -> sources
        for n in notes:
            self.g.add_node(n["path"], title=n["title"])
        for l in links:
            dst = self.resolver.resolve(l["target"], l["src"], l["kind"])
            if dst is None:
                if posixpath.splitext(l["target"])[1].lower() not in ATTACHMENT_EXTS:
                    self.unresolved[l["target"].strip()].append(l["src"])
                continue
            if dst == l["src"]:
                continue
            if self.g.has_edge(l["src"], dst):
                e = self.g[l["src"]][dst]
                e["weight"] += 1
                e["contexts"].append(l["context"])
            else:
                self.g.add_edge(l["src"], dst, weight=1, contexts=[l["context"]], kind=l["kind"])
        self.ug = self.g.to_undirected(as_view=True)
        self._pagerank: dict[str, float] | None = None
        self._clusters: list[set[str]] | None = None

    # ---- basic -----------------------------------------------------------
    def title(self, path: str) -> str:
        return self.notes[path]["title"] if path in self.notes else path

    def backlinks(self, path: str) -> list[dict]:
        out = []
        for src, _, d in self.g.in_edges(path, data=True):
            out.append({"path": src, "title": self.title(src), "count": d["weight"], "contexts": d["contexts"][:3]})
        return sorted(out, key=lambda x: -x["count"])

    def outlinks(self, path: str) -> list[dict]:
        out = []
        for _, dst, d in self.g.out_edges(path, data=True):
            out.append({"path": dst, "title": self.title(dst), "count": d["weight"]})
        return sorted(out, key=lambda x: -x["count"])

    def unresolved_from(self, path: str) -> list[str]:
        return sorted(t for t, srcs in self.unresolved.items() if path in srcs)

    # ---- traversal -------------------------------------------------------
    def neighbors(self, path: str, hops: int = 2, limit: int = 50) -> list[dict]:
        dist = nx.single_source_shortest_path_length(self.ug, path, cutoff=hops)
        paths = nx.single_source_shortest_path(self.ug, path, cutoff=hops)
        pr = self.pagerank()
        out = [
            {"path": p, "title": self.title(p), "hops": d, "via": [self.title(x) for x in paths[p][1:-1]]}
            for p, d in dist.items() if p != path
        ]
        out.sort(key=lambda x: (x["hops"], -pr.get(x["path"], 0)))
        return out[:limit]

    def shortest_path(self, a: str, b: str) -> dict | None:
        try:
            p = nx.shortest_path(self.ug, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        steps = []
        for x, y in zip(p, p[1:]):
            d = self.g.get_edge_data(x, y) or self.g.get_edge_data(y, x)
            direction = "->" if self.g.has_edge(x, y) else "<-"
            steps.append({"from": self.title(x), "to": self.title(y), "direction": direction,
                          "context": d["contexts"][0] if d else ""})
        return {"notes": [{"path": x, "title": self.title(x)} for x in p], "steps": steps}

    # ---- whole-graph analytics ------------------------------------------
    def pagerank(self) -> dict[str, float]:
        if self._pagerank is None:
            self._pagerank = nx.pagerank(self.g, weight="weight") if self.g.number_of_nodes() else {}
        return self._pagerank

    def hubs(self, limit: int = 20, folder: str | None = None) -> list[dict]:
        pr = self.pagerank()
        items = [
            {"path": p, "title": self.title(p), "score": round(s * 1000, 3),
             "backlinks": self.g.in_degree(p), "outlinks": self.g.out_degree(p)}
            for p, s in pr.items() if not folder or p.startswith(folder.strip("/") + "/")
        ]
        return sorted(items, key=lambda x: -x["score"])[:limit]

    def clusters(self, min_size: int = 3) -> list[dict]:
        if self._clusters is None:
            sub = self.ug.subgraph([n for n in self.ug if self.ug.degree(n) > 0])
            self._clusters = (
                list(nx.community.louvain_communities(nx.Graph(sub), weight="weight", seed=42))
                if sub.number_of_nodes() else []
            )
        pr = self.pagerank()
        out = []
        for c in self._clusters:
            if len(c) < min_size:
                continue
            members = sorted(c, key=lambda p: -pr.get(p, 0))
            out.append({"size": len(c), "top_notes": [self.title(p) for p in members[:8]], "paths": members})
        return sorted(out, key=lambda x: -x["size"])

    def cluster_of(self, path: str) -> list[str]:
        self.clusters()
        for c in self._clusters or []:
            if path in c:
                return sorted(c)
        return []

    def orphans(self) -> list[str]:
        return sorted(p for p in self.g if self.g.degree(p) == 0)

    def unresolved_summary(self, limit: int = 50) -> list[dict]:
        items = [{"target": t, "mentions": len(s), "from": sorted(set(s))[:5]} for t, s in self.unresolved.items()]
        return sorted(items, key=lambda x: -x["mentions"])[:limit]
