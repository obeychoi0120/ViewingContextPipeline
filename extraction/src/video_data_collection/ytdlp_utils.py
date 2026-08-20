from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


def ytdlp_base_opts() -> dict[str, object]:
    load_dotenv("config/.env")
    opts: dict[str, object] = {"nocheckcertificate": True}
    cookiefile = existing_cookiefile()
    if cookiefile:
        opts["cookiefile"] = cookiefile

    if shutil.which("deno"):
        opts["js_runtimes"] = {"deno": {}}
        return opts
    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    return opts


def existing_cookiefile() -> str:
    configured = os.getenv("YTDLP_COOKIEFILE", "").strip()
    for value in (configured, "config/cookies.txt"):
        if not value:
            continue
        path = Path(os.path.expanduser(value))
        if path.is_file():
            return str(path)
    return ""
