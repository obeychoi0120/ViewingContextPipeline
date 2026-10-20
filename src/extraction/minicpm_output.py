"""Recover MiniCPM's escaped line breaks without rewriting quoted/code spans."""


def normalize_minicpm_output(text: str) -> str:
    output = []
    index = 0
    closing = None
    quoted = False
    while index < len(text):
        remaining = text[index:]
        if closing is not None:
            if quoted and text[index] == "\\" and index + 1 < len(text):
                output.append(text[index:index + 2])
                index += 2
                continue
            if remaining.startswith(closing):
                output.append(closing)
                index += len(closing)
                closing, quoted = None, False
            else:
                output.append(text[index])
                index += 1
            continue
        marker = next((m for m in ("```", "`", "$$", "$", "\\(", "\\[") if remaining.startswith(m)), None)
        if marker is not None:
            closing = {"\\(": "\\)", "\\[": "\\]"}.get(marker, marker)
            output.append(marker)
            index += len(marker)
            continue
        char = text[index]
        if char == '"' or (char == "'" and (index == 0 or not text[index - 1].isalnum())):
            # Only balanced quotes delimit a string; ordinary apostrophes stay prose.
            if char in text[index + 1:]:
                closing, quoted = char, True
                output.append(char)
                index += 1
                continue
        if char == "\\":
            end = index + 1
            while end < len(text) and text[end] == "\\":
                end += 1
            if end - index > 1:
                output.append(text[index:end])
                index = end
                continue
            escape = next((e for e in ("\\r\\n", "\\n", "\\r") if remaining.startswith(e)), None)
            if escape is not None:
                output.append("\n")
                index += len(escape)
                continue
        output.append(char)
        index += 1
    return "".join(output)
