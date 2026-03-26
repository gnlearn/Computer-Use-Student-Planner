import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from playwright.sync_api import Browser, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


SYSTEM_PROMPT = """You are a web automation agent.

You are given a simplified DOM of a webpage.

Your job:
- Decide the next best action to complete the task
- Use available tools to interact with the page

Rules:
- Only act using tools
- Be step-by-step
- Prefer visible elements
- Use selectors exactly as provided
- If the task is complete, answer with a short completion summary and do not call any tool
"""


DOM_EXTRACTION_SCRIPT = """
() => {
  const interactiveTags = new Set(["a", "button", "input", "select", "textarea", "form"]);

  function isVisible(el) {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return (
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      parseFloat(style.opacity || "1") > 0 &&
      rect.width > 0 &&
      rect.height > 0
    );
  }

  function textOrNull(value) {
    if (!value) return null;
    const t = value.trim();
    return t.length ? t.slice(0, 220) : null;
  }

  function cssPath(el) {
    if (el.id) return `#${CSS.escape(el.id)}`;

    const attrHints = [
      ["name", el.getAttribute("name")],
      ["aria-label", el.getAttribute("aria-label")],
      ["data-testid", el.getAttribute("data-testid")],
      ["placeholder", el.getAttribute("placeholder")]
    ];

    for (const [key, value] of attrHints) {
      if (!value) continue;
      const escaped = CSS.escape(value);
      const selector = `${el.tagName.toLowerCase()}[${key}="${escaped}"]`;
      if (document.querySelectorAll(selector).length === 1) return selector;
    }

    const classes = (el.className || "")
    .split(/\\s+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((cls) => `.${CSS.escape(cls)}`)
      .join("");
    if (classes) {
      const selector = `${el.tagName.toLowerCase()}${classes}`;
      if (document.querySelectorAll(selector).length === 1) return selector;
    }

    let current = el;
    const path = [];
    while (current && current.nodeType === Node.ELEMENT_NODE && current !== document.body) {
      const tag = current.tagName.toLowerCase();
      const siblings = current.parentNode
        ? Array.from(current.parentNode.children).filter((s) => s.tagName === current.tagName)
        : [];
      if (siblings.length > 1) {
        const index = siblings.indexOf(current) + 1;
        path.unshift(`${tag}:nth-of-type(${index})`);
      } else {
        path.unshift(tag);
      }
      const selector = path.join(" > ");
      if (document.querySelectorAll(selector).length === 1) return selector;
      current = current.parentElement;
    }

    return path.join(" > ") || el.tagName.toLowerCase();
  }

  const candidates = Array.from(document.querySelectorAll("a,button,input,select,textarea,form,[role='button']"));
  const output = [];

  for (const el of candidates) {
    const tag = el.tagName.toLowerCase();
    if (!interactiveTags.has(tag) && el.getAttribute("role") !== "button") continue;
    if (!isVisible(el)) continue;

    const item = {
      tag,
      role: textOrNull(el.getAttribute("role")),
      type: textOrNull(el.getAttribute("type")),
      id: textOrNull(el.id),
      name: textOrNull(el.getAttribute("name")),
      placeholder: textOrNull(el.getAttribute("placeholder")),
      label: textOrNull(el.getAttribute("aria-label")),
      text: textOrNull(el.innerText || el.textContent),
      selector: cssPath(el),
      visible: true,
      disabled: el.disabled === true
    };

    output.push(item);
  }

  return output.slice(0, 250);
}
"""


TOOLS = [
    {
        "name": "click_element",
        "description": "Click an element on the page",
        "input_schema": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into an input field",
        "input_schema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["selector", "text"],
        },
    },
    {
        "name": "navigate",
        "description": "Go to a URL",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "submit_form",
        "description": "Submit a form",
        "input_schema": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
    {
        "name": "extract_text",
        "description": "Extract text from element",
        "input_schema": {
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        },
    },
]


def load_dotenv_if_present(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class AgentMemory:
    goal: str
    steps_taken: list[str] = field(default_factory=list)
    current_page: str = ""
    last_action: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps_taken": self.steps_taken[-10:],
            "current_page": self.current_page,
            "last_action": self.last_action,
        }


class BrowserAutomation:
    def __init__(self, headless: bool, timeout_ms: int = 15_000, profile_dir: str | None = None) -> None:
        self._playwright = sync_playwright().start()
        
        if profile_dir:
            # Use persistent context with saved login state
            self.browser_context = self._playwright.chromium.launch_persistent_context(
                profile_dir,
                headless=headless
            )
            self.page: Page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()
            self._browser = None
        else:
            # Use regular browser without saved state
            self._browser = self._playwright.chromium.launch(headless=headless)
            self.browser_context = None
            self.page: Page = self._browser.new_page()
        
        self.page.set_default_timeout(timeout_ms)

    def close(self) -> None:
        try:
            if self.browser_context:
                self.browser_context.close()
            elif self._browser:
                self._browser.close()
        except Exception:
            # Browser/context may already be closed
            pass
        finally:
            try:
                self._playwright.stop()
            except Exception:
                # Playwright may already be stopped
                pass

    def get_state(self) -> dict[str, Any]:
        return {
            "url": self.page.url,
            "title": self.page.title(),
            "dom": self.page.evaluate(DOM_EXTRACTION_SCRIPT),
        }

    def _ensure_selector(self, selector: str) -> None:
        if self.page.query_selector(selector) is None:
            raise ValueError(f"Selector not found on page: {selector}")

    def navigate(self, url: str) -> str:
        self.page.goto(url, wait_until="domcontentloaded")
        return f"Navigated to {self.page.url}"

    def click_element(self, selector: str) -> str:
        self._ensure_selector(selector)
        self.page.click(selector)
        self.page.wait_for_timeout(350)
        return f"Clicked {selector}"

    def type_text(self, selector: str, text: str) -> str:
        self._ensure_selector(selector)
        self.page.fill(selector, text)
        return f"Typed {len(text)} chars into {selector}"

    def submit_form(self, selector: str) -> str:
        self._ensure_selector(selector)
        self.page.eval_on_selector(selector, "(el) => el.submit()")
        self.page.wait_for_timeout(400)
        return f"Submitted form {selector}"

    def extract_text(self, selector: str) -> str:
        self._ensure_selector(selector)
        text = self.page.inner_text(selector).strip()
        return text[:1500]


def call_claude(
    client: Anthropic,
    model: str,
    task: str,
    memory: AgentMemory,
    page_state: dict[str, Any],
    messages: list[dict[str, Any]],
) -> Any:
    user_payload = {
        "task": task,
        "memory": memory.snapshot(),
        "page": {
            "url": page_state.get("url", ""),
            "title": page_state.get("title", ""),
            "interactive_elements": page_state.get("dom", []),
        },
    }
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": json.dumps(user_payload)}],
        }
    )
    return client.messages.create(
        model=model,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )


def execute_tool(browser: BrowserAutomation, name: str, tool_input: dict[str, Any]) -> str:
    if name == "click_element":
        return browser.click_element(tool_input["selector"])
    if name == "type_text":
        return browser.type_text(tool_input["selector"], tool_input["text"])
    if name == "navigate":
        return browser.navigate(tool_input["url"])
    if name == "submit_form":
        return browser.submit_form(tool_input["selector"])
    if name == "extract_text":
        return browser.extract_text(tool_input["selector"])
    raise ValueError(f"Unsupported tool: {name}")


def run_agent(task: str, start_url: str | None, model: str, max_steps: int, headless: bool, timeout_ms: int, profile_dir: str | None = None, setup_mode: bool = False) -> None:
    load_dotenv_if_present()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in environment variables or .env before running this script.")

    browser = BrowserAutomation(headless=headless, timeout_ms=timeout_ms, profile_dir=profile_dir)

    try:
        if start_url:
            print(browser.navigate(start_url))
        
        # In setup mode, just open the browser and wait for user to manually log in
        if setup_mode:
            print("\n[setup mode] Browser is open. You can manually log in now.")
            print("Press Ctrl+C when finished, or this will wait indefinitely.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[setup mode] Login saved. Browser will close.")
                return
        
        client = Anthropic(api_key=api_key)
        memory = AgentMemory(goal=task)
        messages: list[dict[str, Any]] = []

        for step in range(1, max_steps + 1):
            page_state = browser.get_state()
            memory.current_page = page_state.get("url", "")
            response = call_claude(client, model, task, memory, page_state, messages)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            text_blocks = [block.text.strip() for block in response.content if block.type == "text" and block.text.strip()]

            if not tool_uses:
                final_text = "\n".join(text_blocks) if text_blocks else "Task complete (no tool call returned)."
                print(f"\n[complete] {final_text}")
                return

            for tool_use in tool_uses:
                try:
                    result = execute_tool(browser, tool_use.name, tool_use.input)
                    print(f"[step {step}] {tool_use.name} -> {result}")
                except (ValueError, PlaywrightTimeoutError) as exc:
                    result = f"Tool execution error: {type(exc).__name__}: {exc}"
                    print(f"[step {step}] {tool_use.name} -> {result}")

                memory.last_action = f"{tool_use.name}: {tool_use.input}"
                memory.steps_taken.append(memory.last_action)

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": result,
                            }
                        ],
                    }
                )

        print("\n[stopped] Reached max steps before completion.")
    finally:
        browser.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Web Automation Agent (Claude + HTML Control)")
    parser.add_argument("--task", required=True, help="Task goal in plain English")
    parser.add_argument("--start-url", default=None, help="Initial URL to open before the loop")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Claude model name")
    parser.add_argument("--max-steps", type=int, default=30, help="Maximum loop iterations")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--timeout-ms", type=int, default=15000, help="Playwright action timeout in milliseconds")
    parser.add_argument("--profile-dir", default=None, help="Browser profile directory for persistent login state")
    parser.add_argument("--setup-mode", action="store_true", help="Open browser for manual login (requires Ctrl+C to exit)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_agent(
        task=args.task,
        start_url=args.start_url,
        model=args.model,
        max_steps=args.max_steps,
        headless=args.headless,
        timeout_ms=args.timeout_ms,
        profile_dir=args.profile_dir,
        setup_mode=args.setup_mode,
    )


if __name__ == "__main__":
    main()