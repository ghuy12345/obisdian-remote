"""All settings come from environment variables (set them in Coolify)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Vault repo -------------------------------------------------------
    # HTTPS URL with a fine-grained GitHub token that has Contents: read/write
    # on the vault repo only, e.g.
    # https://x-access-token:github_pat_xxx@github.com/owner/vault.git
    vault_repo_url: str = ""
    vault_branch: str = "main"
    data_dir: Path = Path("/data")
    sync_interval_seconds: int = 60

    git_author_name: str = "Vault Brain"
    git_author_email: str = "vault-brain@localhost"

    # --- Auth -------------------------------------------------------------
    # REST API: send "Authorization: Bearer <api_token>".
    api_token: str = ""
    # MCP endpoint lives at https://<host>/<mcp_secret>/mcp. Use a long random
    # string (openssl rand -hex 24). Paste that full URL into Claude as a
    # custom connector.
    mcp_secret: str = ""
    # Optional: public hostname(s) for MCP Host-header checks, comma separated.
    allowed_hosts: str = ""

    # --- Embeddings (any OpenAI-compatible /embeddings endpoint) ----------
    embeddings_base_url: str = "https://openrouter.ai/api/v1"
    embeddings_api_key: str = ""
    embeddings_model: str = "openai/text-embedding-3-small"
    embeddings_batch_size: int = 64

    # --- Indexing ---------------------------------------------------------
    chunk_chars: int = 1500
    chunk_overlap: int = 200
    ignore_dirs: str = ".obsidian,.git,.trash,node_modules"

    # --- Writes (append-only) ---------------------------------------------
    writes_enabled: bool = True
    # Comma-separated folder prefixes new notes may be created in. Empty = anywhere.
    # Appends to existing notes are allowed anywhere unless restricted below.
    write_create_dirs: str = ""
    write_append_dirs: str = ""
    max_write_chars: int = 20000
    # In-place edits (replace text / rewrite a section). Every edit is a git commit,
    # so anything can be restored from history.
    edits_enabled: bool = True
    write_edit_dirs: str = ""
    # Safety net: refuse a single edit that deletes more than this many characters.
    max_edit_delete_chars: int = 5000

    @property
    def vault_dir(self) -> Path:
        return self.data_dir / "vault"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def ignore_dir_set(self) -> set[str]:
        return {d.strip() for d in self.ignore_dirs.split(",") if d.strip()}

    @staticmethod
    def _dirs(value: str) -> list[str]:
        return [d.strip().strip("/") + "/" for d in value.split(",") if d.strip()]

    @property
    def create_dirs(self) -> list[str]:
        return self._dirs(self.write_create_dirs)

    @property
    def append_dirs(self) -> list[str]:
        return self._dirs(self.write_append_dirs)

    @property
    def edit_dirs(self) -> list[str]:
        return self._dirs(self.write_edit_dirs)

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.embeddings_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
