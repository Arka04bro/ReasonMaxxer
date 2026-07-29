import ast
import math
import re

BRACED_COMMANDS = [
    "\\text",
    "\\mathrm",
    "\\mathbf",
    "\\mathit",
    "\\mathtt",
    "\\mathcal",
    "\\operatorname",
    "\\mbox",
    "\\boxed",
]

SIMPLE_REPLACEMENTS = {
    "\\left": "",
    "\\right": "",
    "\\,": "",
    "\\;": "",
    "\\:": "",
    "\\!": "",
    "\\quad": "",
    "\\qquad": "",
    "\\cdot": "*",
    "\\times": "*",
    "\\div": "/",
    "{eq}": "",
    "{/eq}": "",
    "\\Omega": "omega",
    "\\omega": "omega",
}

UNICODE_REPLACEMENTS = {
    "π": "pi",
    "∞": "inf",
    "°": "deg",
    "Ω": "omega",
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "·": "*",
}

# Match standalone numeric tokens; avoid digits embedded in identifiers
# like eq2, x1, a_3, etc. which are common in reasoning/code traces.
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_])")
INTEGER_TOKEN_PATTERN = re.compile(r"-?\d+")
FINAL_MARKER_PATTERN = re.compile(
    r"(?:final answer|the answer is|thus, the answer|therefore, the answer|so the answer is)",
    flags=re.IGNORECASE,
)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)
OPEN_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*)$", flags=re.IGNORECASE | re.DOTALL)


def _extract_braced_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None


def extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(matches):
        group = _extract_braced_group(text, match.end() - 1)
        if group:
            content, _ = group
            return content.strip()
    return None


def extract_answer_tag_answer(text: str) -> str | None:
    if not text:
        return None
    matches = list(ANSWER_TAG_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    match = OPEN_ANSWER_TAG_PATTERN.search(text)
    if not match:
        return None
    candidate = match.group(1)
    # Trim common generation leftovers if stop string was not kept.
    for stop in ("</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>"):
        idx = candidate.find(stop)
        if idx != -1:
            candidate = candidate[:idx]
    candidate = re.split(r"</?(?:think|answer)>", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
    candidate = candidate.strip()
    return candidate or None


def extract_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _canonicalize_numeric_token(token: str, integer_if_close: bool = False) -> str | None:
    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return None
    if INTEGER_TOKEN_PATTERN.fullmatch(cleaned):
        try:
            ivalue = int(cleaned)
        except Exception:
            return None
        return str(ivalue)
    try:
        value = float(cleaned)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    if integer_if_close and abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    # Keep scientific notation / decimals normalized to a stable compact form.
    return f"{value:.15g}"


def extract_final_number(text: str, integer_if_close: bool = False) -> str | None:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    candidates: list[str] = []
    if boxed is not None:
        candidates.extend(NUMBER_PATTERN.findall(boxed))
    candidates.extend(NUMBER_PATTERN.findall(text))
    for token in reversed(candidates):
        normalized = _canonicalize_numeric_token(token, integer_if_close=integer_if_close)
        if normalized is not None:
            return normalized
    return None


def extract_final_integer(text: str, min_value: int | None = None, max_value: int | None = None) -> str | None:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    candidates: list[str] = []
    if boxed is not None:
        candidates.extend(NUMBER_PATTERN.findall(boxed))
    candidates.extend(NUMBER_PATTERN.findall(text))
    if not candidates:
        return None

    for token in reversed(candidates):
        normalized = _canonicalize_numeric_token(token, integer_if_close=True)
        if normalized is None or not re.fullmatch(r"-?\d+", normalized):
            continue
        value = int(normalized)
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        return str(value)
    return None


def extract_multiple_choice_letter(text: str) -> str | None:
    if not text:
        return None

    boxed = extract_boxed_answer(text)
    if boxed:
        match = re.search(r"\b([A-Ea-e])\b", boxed)
        if match:
            return match.group(1).upper()

    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?\s*([A-Ea-e])\s*\)?",
        r"\(([A-Ea-e])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()

    line_tail = text.strip().splitlines()[-1] if text.strip() else ""
    match = re.fullmatch(r"\(?\s*([A-Ea-e])\s*\)?\.?", line_tail)
    if match:
        return match.group(1).upper()
    return None


def _unwrap_python_list_singleton(text: str) -> str | None:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return None
    if not isinstance(parsed, list) or len(parsed) == 0:
        return None
    # Keep deterministic behavior: if multiple references appear, use the first.
    first = parsed[0]
    return str(first).strip() if first is not None else None


def _extract_final_line_candidate(text: str) -> str | None:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    for line in reversed(lines):
        lower = line.lower()
        if "final answer" in lower or lower.startswith("answer"):
            candidate = re.sub(
                r"^.*?(?:final answer(?:\s+is)?|answer(?:\s+is)?|answer:)\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()
            candidate = candidate.strip(". ")
            if candidate:
                return candidate

    tail = lines[-1].strip(". ")
    if tail and len(tail) <= 120:
        return tail
    return None


def _extract_final_answer_region(text: str, max_window: int = 420) -> str | None:
    if not text:
        return None
    markers = list(FINAL_MARKER_PATTERN.finditer(text))
    if not markers:
        return None
    start = markers[-1].start()
    region = text[start : start + max_window]
    if not region:
        return None

    # Trim likely prompt-restart artifacts where the model starts solving
    # a new unrelated problem after already stating an answer.
    cut = len(region)
    restart_tokens = [" Given ", " If ", " Let ", " Problem:", " Question:"]
    for token in restart_tokens:
        pos = region.find(token, 8)
        if pos != -1:
            cut = min(cut, pos)
    return region[:cut].strip() or None


def _cleanup_math_candidate(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    c = candidate.strip()
    if not c:
        return None
    # Remove common trailing punctuation/noise around copied answer snippets.
    c = c.strip(" .,:;")
    if not c:
        return None
    # Remove common math delimiters if present.
    c = c.strip()
    if c.startswith("$$") and c.endswith("$$") and len(c) > 4:
        c = c[2:-2].strip()
    c = c.strip("$").strip()
    if not c:
        return None
    # If candidate is an equation, keep the final RHS expression.
    # Example: "(0,3) = (3,\\frac{\\pi}{2})" -> "(3,\\frac{\\pi}{2})"
    if "=" in c:
        rhs = c.split("=")[-1].strip()
        if rhs:
            c = rhs
    # Drop wrappers like "Answer: ...".
    c = re.sub(r"^(?:answer(?:\s+is)?|final answer(?:\s+is)?|thus|therefore)\s*[:\-]?\s*", "", c, flags=re.IGNORECASE)
    c = c.strip(" .,:;")
    return c or None


def extract_math500_answer_flexible(text: str) -> str | None:
    if not text:
        return None

    tagged = extract_answer_tag_answer(text)
    if tagged is not None:
        return tagged

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    region = _extract_final_answer_region(text)
    sources = [src for src in [region, text] if src]

    # Shared helpers for steps 1–3.
    # _PROSE_WORD_RE: 3+ consecutive letters not preceded by backslash.
    # Matches English words like "for", "the", "correct", "double" but NOT LaTeX commands
    # like \frac, \sqrt, \le (those are preceded by \).
    _PROSE_WORD_RE = re.compile(r"(?<!\\)[A-Za-z]{3,}")
    _HAS_MATH_CHAR = re.compile(r"[\d\\\{\}\|]")

    # 1) Last inline/display math segment.
    # Filters applied to each candidate before accepting:
    #  a) Skip multi-line spans — those are adjacent dollar-amount pairs like "$20...$40",
    #     not genuine LaTeX inline math.
    #  b) Skip content with English prose words (3+ consecutive letters not preceded by \) —
    #     catches "2.50 for the stamp, which is double of" from $2.50...$1.25 spans.
    #  c) Require at least one digit — skips pure-variable expressions like "x = y" or
    #     "x \le y" from Note sections that are valid LaTeX but not numeric answers.
    math_seg_pattern = re.compile(r"\$(.+?)\$", flags=re.S)
    for src in sources:
        matches = math_seg_pattern.findall(src)
        for m in reversed(matches):
            if "\n" in m:
                continue
            if _PROSE_WORD_RE.search(m):
                continue
            if not re.search(r"\d", m):
                continue
            cand = _cleanup_math_candidate(m)
            if cand:
                return cand

    # 2) Last short parenthesized expression (useful for coordinate answers).
    # Require the opening paren is NOT immediately preceded by a word character so that
    # function-argument groups like sqrt(29) or f(x) are excluded.
    tuple_pattern = re.compile(r"(?<![A-Za-z_0-9])\(([^()\n]{1,120})\)")
    for src in sources:
        matches = tuple_pattern.findall(src)
        for m in reversed(matches):
            cand = _cleanup_math_candidate(f"({m})")
            if cand:
                return cand

    # 3) Last explicit equals-clause on a line.
    # Filters on raw match: must contain a digit/math symbol (rejects "is correct", "is equivalent to").
    # Filters on cleaned result: must not contain prose words (rejects "double of $1.25") and
    # must not still contain "is" as a word (rejects "9 is 5" from "x^2 = 9 is 5").
    eq_pattern = re.compile(r"(?:=|is)\s*([^\n]{1,140})")
    for src in sources:
        matches = eq_pattern.findall(src)
        for m in reversed(matches):
            if not _HAS_MATH_CHAR.search(m):
                continue
            cand = _cleanup_math_candidate(m)
            if cand and _HAS_MATH_CHAR.search(cand):
                if re.search(r"\bis\b", cand):
                    continue
                if _PROSE_WORD_RE.search(cand):
                    continue
                return cand

    # 4) Try final-line style candidate after structured extraction patterns.
    # Require the candidate to contain at least one digit/math char and no prose words —
    # otherwise fall through to numeric extraction.
    for src in sources:
        candidate = _extract_final_line_candidate(src)
        candidate = _cleanup_math_candidate(candidate)
        if candidate and _HAS_MATH_CHAR.search(candidate):
            if _PROSE_WORD_RE.search(candidate):
                continue
            boxed_candidate = extract_boxed_answer(candidate)
            if boxed_candidate is not None:
                return boxed_candidate
            return candidate

    # 5) Fallback to numeric extraction.
    number = extract_final_number(text, integer_if_close=False)
    if number is not None:
        return number
    return None


def get_ground_truth(solution_text: str, dataset: str = "math500") -> str | None:
    if dataset == "math500":
        return extract_boxed_answer(solution_text)
    if dataset == "gsm8k":
        return extract_gsm8k_answer(solution_text)
    if dataset == "aime24":
        return extract_final_integer(solution_text, min_value=0, max_value=999)
    if dataset == "amc23":
        letter = extract_multiple_choice_letter(solution_text)
        if letter is not None:
            return letter
        number = extract_final_number(solution_text, integer_if_close=True)
        if number is not None:
            return number
        return solution_text.strip() or None
    if dataset in {"minerva_math", "olympiadbench"}:
        text = solution_text
        if dataset == "olympiadbench":
            unwrapped = _unwrap_python_list_singleton(text)
            if unwrapped is not None:
                text = unwrapped
        boxed = extract_boxed_answer(text)
        if boxed is not None:
            return boxed
        return text.strip() or None
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_answer(text: str, dataset: str = "math500") -> str | None:
    if dataset == "math500":
        tagged = extract_answer_tag_answer(text)
        if tagged is not None:
            return tagged
        return extract_boxed_answer(text)
    if dataset == "gsm8k":
        tagged = extract_answer_tag_answer(text)
        if tagged is not None:
            return tagged
        return extract_gsm8k_answer(text)
    if dataset == "aime24":
        tagged = extract_answer_tag_answer(text)
        if tagged is not None:
            parsed = extract_final_integer(tagged, min_value=0, max_value=999)
            if parsed is not None:
                return parsed
        region = _extract_final_answer_region(text)
        if region:
            boxed_region = extract_boxed_answer(region)
            if boxed_region is not None:
                parsed = extract_final_integer(boxed_region, min_value=0, max_value=999)
                if parsed is not None:
                    return parsed
            parsed = extract_final_integer(region, min_value=0, max_value=999)
            if parsed is not None:
                return parsed
        return extract_final_integer(text, min_value=0, max_value=999)
    if dataset == "amc23":
        tagged = extract_answer_tag_answer(text)
        if tagged is not None:
            letter_tag = extract_multiple_choice_letter(tagged)
            if letter_tag is not None:
                return letter_tag
            parsed_tag = extract_final_number(tagged, integer_if_close=True)
            if parsed_tag is not None:
                return parsed_tag
        region = _extract_final_answer_region(text)
        if region:
            letter_region = extract_multiple_choice_letter(region)
            if letter_region is not None:
                return letter_region
            parsed_region = extract_final_number(region, integer_if_close=True)
            if parsed_region is not None:
                return parsed_region
        letter = extract_multiple_choice_letter(text)
        if letter is not None:
            return letter
        return extract_final_number(text, integer_if_close=True)
    if dataset in {"minerva_math", "olympiadbench"}:
        tagged = extract_answer_tag_answer(text)
        if tagged is not None:
            return tagged
        boxed = extract_boxed_answer(text)
        if boxed is not None:
            return boxed
        candidate = _extract_final_line_candidate(text)
        if candidate:
            boxed_candidate = extract_boxed_answer(candidate)
            if boxed_candidate is not None:
                return boxed_candidate
            return candidate
        number = extract_final_number(text, integer_if_close=False)
        if number is not None:
            return number
        return extract_gsm8k_answer(text)
    raise ValueError(f"Unsupported dataset: {dataset}")


def extract_answer_with_mode(
    text: str,
    dataset: str = "math500",
    *,
    math500_extractor: str = "boxed",
) -> str | None:
    if dataset == "math500":
        mode = (math500_extractor or "boxed").strip().lower()
        if mode == "boxed":
            return extract_boxed_answer(text)
        if mode == "flexible":
            return extract_math500_answer_flexible(text)
        raise ValueError("math500_extractor must be one of: boxed, flexible")
    return extract_answer(text, dataset=dataset)


def _replace_braced_commands(text: str) -> str:
    i = 0
    out = []
    while i < len(text):
        if text[i] == "\\":
            matched = False
            for cmd in BRACED_COMMANDS:
                if text.startswith(cmd, i):
                    j = i + len(cmd)
                    while j < len(text) and text[j].isspace():
                        j += 1
                    if j < len(text) and text[j] == "{":
                        group = _extract_braced_group(text, j)
                        if group:
                            content, end = group
                            out.append(content)
                            i = end
                            matched = True
                            break
            if matched:
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _replace_frac(text: str) -> str:
    i = 0
    out = []
    frac_commands = ("\\frac", "\\dfrac", "\\tfrac", "\\cfrac")
    while i < len(text):
        matched_cmd = next((cmd for cmd in frac_commands if text.startswith(cmd, i)), None)
        if matched_cmd is not None:
            j = i + len(matched_cmd)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                num_group = _extract_braced_group(text, j)
                if num_group:
                    num, end_num = num_group
                    k = end_num
                    while k < len(text) and text[k].isspace():
                        k += 1
                    if k < len(text) and text[k] == "{":
                        den_group = _extract_braced_group(text, k)
                        if den_group:
                            den, end_den = den_group
                            out.append(f"({num})/({den})")
                            i = end_den
                            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _replace_sqrt(text: str) -> str:
    i = 0
    out = []
    while i < len(text):
        if text.startswith("\\sqrt", i):
            j = i + len("\\sqrt")
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                group = _extract_braced_group(text, j)
                if group:
                    content, end = group
                    out.append(f"sqrt({content})")
                    i = end
                    continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _replace_unicode_sqrt(text: str) -> str:
    text = re.sub(r"√\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"√\s*\(([^()]+)\)", r"sqrt(\1)", text)
    text = re.sub(r"√\s*([A-Za-z0-9_.]+)", r"sqrt(\1)", text)
    return text


def _strip_single_equation_wrapper(text: str) -> str:
    if text.count("=") != 1:
        return text
    lhs, rhs = text.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not rhs:
        return text
    if not lhs:
        return rhs
    if len(rhs) < len(text):
        return rhs
    return text


def _insert_implicit_multiplication(text: str) -> str:
    updated = text
    patterns = [
        (r"(\d)(pi|sqrt\(|\()", r"\1*\2"),
        (r"(\))(pi|sqrt\(|\d|\()", r"\1*\2"),
        (r"(pi)(\d|sqrt\(|\()", r"\1*\2"),
    ]
    for pattern, repl in patterns:
        updated = re.sub(pattern, repl, updated)
    return updated


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = answer.strip()
    if not text:
        return None

    text = text.replace("$", "")
    text = text.replace("\\n", " ").replace("\n", " ")

    for key, value in UNICODE_REPLACEMENTS.items():
        text = text.replace(key, value)

    text = _replace_braced_commands(text)
    text = _replace_frac(text)
    text = _replace_sqrt(text)
    text = _replace_unicode_sqrt(text)

    for key, value in SIMPLE_REPLACEMENTS.items():
        text = text.replace(key, value)

    text = text.replace("\\pi", "pi")
    text = text.replace("\\infty", "inf")
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)

    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")

    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _strip_single_equation_wrapper(text)
    text = _insert_implicit_multiplication(text)
    text = text.strip(".")
    return text
