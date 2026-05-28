"""Python code execution tool for AMIA agents."""

import sys, io, traceback
from langchain_core.tools import tool


@tool
def run_python(code: str) -> str:
    """
    Execute Python code and return the output.

    Use for calculations, data processing, or analysis.
    Always use print() to produce output.

    Args:
        code: Valid Python code to execute.
    """
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {})
        output = buffer.getvalue()
        return output or "Code executed successfully (no output printed)."
    except Exception:
        return f"Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout