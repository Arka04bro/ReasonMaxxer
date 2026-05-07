import ast
import math
import re
import warnings
from typing import Any

from reasonmaxxer.answer_extraction import normalize_answer


class _SafeEval(ast.NodeVisitor):
    def __init__(self) -> None:
        self._allowed_names = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "inf": math.inf,
        }

    def visit(self, node: ast.AST) -> Any:
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function")
        func_name = node.func.id
        if func_name not in self._allowed_names:
            raise ValueError("Unsupported function")
        func = self._allowed_names[func_name]
        if len(node.args) != 1:
            raise ValueError("Unsupported number of args")
        return func(self.visit(node.args[0]))

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self._allowed_names:
            return self._allowed_names[node.id]
        raise ValueError("Unsupported name")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported constant")

    def visit_Num(self, node: ast.Num) -> Any:
        return node.n


def _safe_eval_numeric(expr: str) -> float | None:
    if "," in expr:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.parse(expr, mode="eval")
        return float(_SafeEval().visit(parsed))
    except Exception:
        return None


def _parse_quantity(expr: str) -> tuple[float, str | None] | None:
    cleaned = expr.strip()
    match = re.fullmatch(
        r"\(?\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)?(?:\*?\s*(deg|rad|omega))?",
        cleaned,
    )
    if not match:
        return None
    try:
        value = float(match.group(1))
    except Exception:
        return None
    unit = match.group(2)
    return value, unit


def _quantity_match(pred_norm: str, gt_norm: str, tol: float) -> bool | None:
    pred_q = _parse_quantity(pred_norm)
    gt_q = _parse_quantity(gt_norm)
    if pred_q is None or gt_q is None:
        return None

    pred_val, pred_unit = pred_q
    gt_val, gt_unit = gt_q

    compatible_nonangular = {None, "omega"}
    if {pred_unit, gt_unit}.issubset(compatible_nonangular):
        return abs(pred_val - gt_val) <= tol

    if pred_unit == gt_unit and pred_unit in {"deg", "rad"}:
        return abs(pred_val - gt_val) <= tol

    if {pred_unit, gt_unit} == {"deg", "rad"}:
        if pred_unit == "deg":
            pred_val = math.radians(pred_val)
        if gt_unit == "deg":
            gt_val = math.radians(gt_val)
        # Unit-converted answers are often rounded; allow a small relative tolerance.
        rel_tol = 1e-3
        return abs(pred_val - gt_val) <= max(tol, abs(gt_val) * rel_tol)

    return None


def _split_ground_truth_candidates(ground_truth: str | None) -> list[str]:
    if ground_truth is None:
        return []
    text = str(ground_truth).strip()
    if not text:
        return []

    candidates: list[str] = [text]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            parsed_candidates = [str(item).strip() for item in parsed if str(item).strip()]
            if parsed_candidates:
                candidates = parsed_candidates

    if "||" in text:
        candidates.extend(part.strip() for part in text.split("||") if part.strip())

    # De-duplicate while keeping order.
    dedup: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _answers_match_single(predicted: str | None, ground_truth: str | None, tol: float) -> bool:
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm is None or gt_norm is None:
        return False

    if pred_norm == gt_norm:
        return True

    quantity_match = _quantity_match(pred_norm, gt_norm, tol)
    if quantity_match is not None:
        return quantity_match

    pred_val = _safe_eval_numeric(pred_norm)
    gt_val = _safe_eval_numeric(gt_norm)
    if pred_val is not None and gt_val is not None:
        return abs(pred_val - gt_val) <= tol

    try:
        import sympy as sp

        pred_expr = sp.sympify(pred_norm)
        gt_expr = sp.sympify(gt_norm)
        return sp.simplify(pred_expr - gt_expr) == 0
    except Exception:
        return False


def answers_match(predicted: str | None, ground_truth: str | None, tol: float = 1e-6) -> bool:
    if predicted is None or ground_truth is None:
        return False
    candidates = _split_ground_truth_candidates(ground_truth)
    if not candidates:
        return False
    for candidate in candidates:
        if _answers_match_single(predicted, candidate, tol=tol):
            return True
    return False
