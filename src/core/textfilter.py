"""
filter_text_lines.py — strip low-value/sensitive lines from a raw one-
message-per-line text export before it goes into persona_forge's distiller.

Drops lines that:
  - contain a URL (http(s)://, www., or a bare domain like example.com)
  - contain a phone-number-shaped sequence
  - are shorter than --min-chars after stripping (default 12)

Reports a before/after count so you can see how much survives before
deciding on a --sample size for `persona_forge.py build`.

Usage:
    python filter_text_lines.py "training_input/bob/user_responses (2).txt" -o "training_input/bob/user_responses_filtered.txt"
"""

import re
import sys
import argparse

URL_RE = re.compile(
    r"(https?://\S+)"
    r"|(\bwww\.\S+)"
    r"|(\b[a-zA-Z0-9-]+\.(?:com|net|org|io|gg|tv|co|me|xyz|app|dev|gov|edu|ai)\b\S*)",
    re.IGNORECASE,
)

# Requires an actual separator between digit groups — 555-123-4567,
# (555) 123-4567, 555.123.4567, +1 555 123 4567. A bare unbroken 10-digit
# run is deliberately NOT matched: that swept up Discord snowflake IDs, game
# seeds, and other incidental big numbers as false positives in testing.
PHONE_RE = re.compile(
    r"(\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)

# Discord mention/channel/role tags, and bare snowflake IDs (15-20 digit
# runs — the raw ID Discord uses under the hood, sometimes pasted without
# the <@...> wrapper) — stripped out, not a drop reason on their own, since
# good content often sits right next to one.
MENTION_RE = re.compile(r"<@!?\d+>|<#\d+>|<@&\d+>|\b\d{15,20}\b")


def should_drop(line, min_chars):
    stripped = MENTION_RE.sub("", line).strip().rstrip(",").strip()
    if len(stripped) < min_chars:
        return "too_short", stripped
    if URL_RE.search(stripped):
        return "url", stripped
    if PHONE_RE.search(stripped):
        return "phone", stripped
    return None, stripped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_path")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--min-chars", type=int, default=12)
    args = ap.parse_args()

    counts = {"total": 0, "too_short": 0, "url": 0, "phone": 0, "kept": 0}
    kept_lines = []

    with open(args.input_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            counts["total"] += 1
            reason, cleaned = should_drop(line, args.min_chars)
            if reason:
                counts[reason] += 1
            else:
                counts["kept"] += 1
                kept_lines.append(cleaned)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines) + "\n")

    print(f"Total lines read:     {counts['total']:>8}")
    print(f"Dropped — too short:  {counts['too_short']:>8}  (< {args.min_chars} chars)")
    print(f"Dropped — URL:        {counts['url']:>8}")
    print(f"Dropped — phone:      {counts['phone']:>8}")
    print(f"Kept:                 {counts['kept']:>8}")
    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
