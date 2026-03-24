import os
import subprocess
import time
import base64
from pathlib import Path

import anthropic

try:
    import pyautogui
except Exception:
    pyautogui = None


def _run_bash_tool(tool_input: dict) -> str:
    command = tool_input.get("command") or tool_input.get("cmd")
    if not command:
        return "Error: bash tool missing 'command'."

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        return (
            f"exit_code={completed.returncode}\n"
            f"stdout:\n{stdout if stdout else '<empty>'}\n"
            f"stderr:\n{stderr if stderr else '<empty>'}"
        )
    except subprocess.TimeoutExpired:
        return "Error: bash command timed out after 60 seconds."
    except Exception as exc:
        return f"Error running bash command: {type(exc).__name__}: {exc}"


def _run_text_editor_tool(tool_input: dict) -> str:
    command = tool_input.get("command")
    path_value = tool_input.get("path")

    if not command:
        return "Error: text editor tool missing 'command'."
    if not path_value:
        return "Error: text editor tool missing 'path'."

    path = Path(path_value).expanduser().resolve()

    try:
        if command == "view":
            if not path.exists():
                return f"Error: file does not exist: {path}"
            try:
                content = path.read_text(encoding="utf-8")
                return content[:12000]
            except UnicodeDecodeError:
                size = path.stat().st_size
                return f"Binary file: {path} ({size} bytes). Use an image/file viewer instead of text view."

        if command == "create":
            file_text = tool_input.get("file_text", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(file_text, encoding="utf-8")
            return f"Created file: {path}"

        if command == "str_replace":
            if not path.exists():
                return f"Error: file does not exist: {path}"

            old_str = tool_input.get("old_str")
            new_str = tool_input.get("new_str", "")
            if old_str is None:
                return "Error: str_replace needs 'old_str'."

            content = path.read_text(encoding="utf-8")
            occurrences = content.count(old_str)
            if occurrences == 0:
                return "Error: old_str not found."
            if occurrences > 1:
                return "Error: old_str appears multiple times; be more specific."

            updated = content.replace(old_str, new_str, 1)
            path.write_text(updated, encoding="utf-8")
            return f"Replaced text once in: {path}"

        if command == "insert":
            if not path.exists():
                return f"Error: file does not exist: {path}"

            insert_line = tool_input.get("insert_line")
            new_str = tool_input.get("new_str", "")
            if insert_line is None:
                return "Error: insert needs 'insert_line'."

            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            idx = int(insert_line)
            if idx < 0 or idx > len(lines):
                return f"Error: insert_line out of range (0..{len(lines)})."

            lines.insert(idx, new_str)
            path.write_text("".join(lines), encoding="utf-8")
            return f"Inserted text at line {idx} in: {path}"

        return f"Error: unsupported text editor command '{command}'."

    except Exception as exc:
        return f"Error running text editor command: {type(exc).__name__}: {exc}"


def _screenshot_tool_result() -> dict:
    if pyautogui is None:
        return {
            "type": "text",
            "text": "Error: pyautogui is not installed. Install it with: pip install pyautogui",
        }

    image = pyautogui.screenshot()
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64_data,
        },
    }


def _run_computer_tool(tool_input: dict):
    if pyautogui is None:
        return "Error: pyautogui is not installed. Install it with: pip install pyautogui"

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    action = tool_input.get("action")
    if not action:
        return "Error: computer tool missing 'action'."

    try:
        if action == "screenshot":
            return _screenshot_tool_result()

        if action == "mouse_move":
            x = int(tool_input.get("x"))
            y = int(tool_input.get("y"))
            pyautogui.moveTo(x, y, duration=0.1)
            return f"Moved mouse to ({x}, {y})."

        if action == "left_click":
            x = tool_input.get("x")
            y = tool_input.get("y")
            if x is not None and y is not None:
                pyautogui.click(int(x), int(y), button="left")
            else:
                pyautogui.click(button="left")
            return "Left click performed."

        if action == "right_click":
            x = tool_input.get("x")
            y = tool_input.get("y")
            if x is not None and y is not None:
                pyautogui.click(int(x), int(y), button="right")
            else:
                pyautogui.click(button="right")
            return "Right click performed."

        if action == "middle_click":
            x = tool_input.get("x")
            y = tool_input.get("y")
            if x is not None and y is not None:
                pyautogui.click(int(x), int(y), button="middle")
            else:
                pyautogui.click(button="middle")
            return "Middle click performed."

        if action == "double_click":
            x = tool_input.get("x")
            y = tool_input.get("y")
            if x is not None and y is not None:
                pyautogui.doubleClick(int(x), int(y))
            else:
                pyautogui.doubleClick()
            return "Double click performed."

        if action == "left_click_drag":
            start_x = int(tool_input.get("start_x", tool_input.get("x")))
            start_y = int(tool_input.get("start_y", tool_input.get("y")))
            end_x = int(tool_input.get("end_x"))
            end_y = int(tool_input.get("end_y"))
            pyautogui.moveTo(start_x, start_y, duration=0.05)
            pyautogui.dragTo(end_x, end_y, duration=0.2, button="left")
            return f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})."

        if action == "scroll":
            amount = int(tool_input.get("scroll_amount", tool_input.get("amount", 0)))
            x = tool_input.get("x")
            y = tool_input.get("y")
            if x is not None and y is not None:
                pyautogui.scroll(amount, x=int(x), y=int(y))
            else:
                pyautogui.scroll(amount)
            return f"Scrolled by {amount}."

        if action == "key":
            key = tool_input.get("text") or tool_input.get("key")
            if not key:
                return "Error: key action requires 'text' or 'key'."
            if isinstance(key, str) and "+" in key:
                keys = [k.strip().lower() for k in key.split("+") if k.strip()]
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(str(key).lower())
            return f"Pressed key(s): {key}"

        if action == "type":
            text = tool_input.get("text", "")
            pyautogui.write(str(text), interval=0.01)
            return f"Typed {len(str(text))} characters."

        if action == "wait":
            seconds = float(tool_input.get("seconds", 1))
            seconds = max(0.0, min(seconds, 10.0))
            time.sleep(seconds)
            return f"Waited {seconds} second(s)."

        if action == "cursor_position":
            x, y = pyautogui.position()
            return f"Cursor at ({x}, {y})."

        return f"Error: unsupported computer action '{action}'."
    except Exception as exc:
        return f"Error running computer action '{action}': {type(exc).__name__}: {exc}"


def run_tool_locally(name: str, tool_input: dict):
    if name == "bash":
        return _run_bash_tool(tool_input)

    if name == "str_replace_based_edit_tool":
        return _run_text_editor_tool(tool_input)

    if name == "computer":
        return _run_computer_tool(tool_input)

    return f"Error: unsupported tool '{name}'."


def _as_tool_result_content_blocks(result):
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    return [{"type": "text", "text": str(result)}]


def _preview_result_for_log(result) -> str:
    if isinstance(result, str):
        return result[:500]
    if isinstance(result, dict):
        result_type = result.get("type", "unknown")
        if result_type == "image":
            return "<image block returned>"
        return str(result)[:500]
    if isinstance(result, list):
        return f"<list result with {len(result)} block(s)>"
    return str(result)[:500]


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in this terminal. "
            "In PowerShell run: $env:ANTHROPIC_API_KEY='your_key_here'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    tools = [
        {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": 1920,
            "display_height_px": 1080,
            "display_number": 1,
        },
        {
            "type": "text_editor_20250728",
            "name": "str_replace_based_edit_tool",
        },
        {
            "type": "bash_20250124",
            "name": "bash",
        },
    ]

    messages = [{"role": "user", "content": "Using outlook send an email to nico.gonnella@drake.edu and say hello! with the subject Computer use. Use the mouse and keybord shortcuts like tab and ctrl enter to navigate the email interface."}]

    while True:
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            tools=tools,
            messages=messages,
            betas=["computer-use-2025-11-24"],
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            text_blocks = [block.text for block in response.content if block.type == "text"]
            print("\n".join(text_blocks) if text_blocks else response)
            break

        for tool_use in tool_uses:
            print(f"[tool] {tool_use.name}: {tool_use.input}")
            result = run_tool_locally(tool_use.name, tool_use.input)
            print(f"[tool_result] {_preview_result_for_log(result)}\n")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": _as_tool_result_content_blocks(result),
                        }
                    ],
                }
            )


if __name__ == "__main__":
    main()