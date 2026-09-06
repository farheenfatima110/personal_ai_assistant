"""Safe arithmetic calculator.

Parses the expression into an AST and evaluates only a small whitelist of
numeric operations - no ``eval``, no builtins, no attribute access.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "min": min,
    "max": max,
}
_NAMES = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _FUNCS:
            raise ValueError(f"Function '{node.func.id}' is not allowed.")
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported expression.")


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="Arithmetic expression, e.g. '25 * 4 + 10' or 'sqrt(144) + 2**3'",
    )


class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = (
        "Evaluates arithmetic expressions safely. Supports + - * / // % **, "
        "parentheses, and functions: sqrt, abs, round, floor, ceil, log, log10, "
        "exp, sin, cos, tan, min, max, plus constants pi and e."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            result = _eval(tree)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"Result: {result}"
        except ZeroDivisionError:
            return "Calculation error: division by zero."
        except Exception as exc:  # noqa: BLE001 - surface a readable message to the agent
            return f"Calculation error: {exc}"
