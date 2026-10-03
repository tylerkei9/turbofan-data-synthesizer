"""Embed regenerated data into the dashboard page.

dashboard/index.html is a self-contained "bundled" page: the whole page is stored as one JSON-encoded
string inside a <script type="__bundler/template"> block. This script decodes that string, replaces
the three data blocks with the current files next to it, and writes the page back in the same format:

    window.CM       <- dashboard/demo_data.js   (bundled rows, predictions, comparison)
    window.CKPT     <- dashboard/ckpt_meta.json  (checkpoint metadata)
    window.REPLAYS  <- dashboard/replays.json    (recorded training runs)

Usage (from the repository root):  python dashboard/embed_data.py
Regenerate the three files first with api/record_replays.py (see docs/running-the-code.md).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"


def encode(text: str) -> str:
    return json.dumps(text, ensure_ascii=False).replace("</", "<\\u002F")


def main() -> None:
    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r'<script type="__bundler/template">(.*?)</script>', html, re.S)
    raw = m.group(1)
    lead, trail = raw[:len(raw) - len(raw.lstrip())], raw[len(raw.rstrip()):]
    page = json.loads(raw)
    assert lead + encode(page) + trail == raw, "unexpected page encoding; not modifying it"

    cm = (HERE / "demo_data.js").read_text(encoding="utf-8").strip()
    ckpt = json.dumps(json.loads((HERE / "ckpt_meta.json").read_text(encoding="utf-8")), separators=(",", ":"))
    replays = (HERE / "replays.json").read_text(encoding="utf-8").strip()

    page, n1 = re.subn(r"<script>\nwindow\.CM=.*?\n</script>", lambda _: "<script>\n" + cm + "\n</script>", page, count=1, flags=re.S)
    page, n2 = re.subn(r"window\.CKPT = \{.*?\};\n</script>", lambda _: "window.CKPT = " + ckpt + ";\n</script>", page, count=1, flags=re.S)
    page, n3 = re.subn(r"window\.REPLAYS = \{.*?\};\n</script>", lambda _: "window.REPLAYS = " + replays + ";\n</script>", page, count=1, flags=re.S)
    assert (n1, n2, n3) == (1, 1, 1), f"data blocks not found: CM={n1} CKPT={n2} REPLAYS={n3}"

    PAGE.write_text(html[:m.start(1)] + lead + encode(page) + trail + html[m.end(1):], encoding="utf-8")
    print(f"embedded demo_data.js, ckpt_meta.json and replays.json into {PAGE.name}")


if __name__ == "__main__":
    main()
