import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _load_key_value_env_file(file_path: Path) -> None:
    if not file_path.exists():
        return

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:]

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def load_local_environment(search_root: Path | None = None) -> None:
    root = search_root or Path(__file__).resolve().parents[1]
    candidate_files: Iterable[Path] = (
        root / ".env",
        root / ".envrc",
        root / "agent_architecture" / ".env",
    )

    for candidate_file in candidate_files:
        _load_key_value_env_file(candidate_file)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    openai_api_key: str
    openai_model: str
    openai_temperature: float

    @classmethod
    def from_env(cls, load_files: bool = True) -> "AgentRuntimeConfig":
        if load_files:
            load_local_environment()

        api_key = os.environ.get("OPENAI_API_KEY", "")

        return cls(
            openai_api_key=api_key,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            openai_temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
        )
