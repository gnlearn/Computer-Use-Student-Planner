import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic, RateLimitError
from playwright.sync_api import Browser, BrowserContext, Error as PlaywrightError, Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


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


MAX_DOM_ELEMENTS = 80
MAX_HISTORY_MESSAGES = 12
MAX_CLAUDE_RETRIES = 3


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
    def __init__(
        self,
        headless: bool,
        timeout_ms: int = 15_000,
        profile_dir: str | None = None,
        use_personal_edge: bool = False,
        edge_user_data_dir: str | None = None,
        edge_profile_dir: str = "Default",
        keep_open: bool = False,
    ) -> None:
        self._playwright = sync_playwright().start()
        self._keep_open = keep_open

        if use_personal_edge:
            if not edge_user_data_dir:
                local_app_data = os.getenv("LOCALAPPDATA")
                if not local_app_data:
                    raise RuntimeError("LOCALAPPDATA is not set; provide --edge-user-data-dir.")
                edge_user_data_dir = str(Path(local_app_data) / "Microsoft" / "Edge" / "User Data")

            # Personal profile mode must be non-headless and uses your existing Edge profile data.
            self.browser_context: BrowserContext = self._playwright.chromium.launch_persistent_context(
                user_data_dir=edge_user_data_dir,
                channel="msedge",
                headless=False,
                args=[f"--profile-directory={edge_profile_dir}"],
            )
            self.page: Page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()
            self._browser = None
        elif profile_dir:
            # Persistent profile mode stores login state in a reusable local folder.
            profile_path = Path(profile_dir)
            profile_path.mkdir(parents=True, exist_ok=True)
            self.browser_context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                channel="msedge",
                headless=headless,
            )
            self.page = self.browser_context.pages[0] if self.browser_context.pages else self.browser_context.new_page()
            self._browser = None
        else:
            # Use regular browser without saved state
            self._browser = self._playwright.chromium.launch(channel="msedge", headless=headless)
            self.browser_context = None
            self.page: Page = self._browser.new_page()
        
        self.page.set_default_timeout(timeout_ms)

    def close(self) -> None:
        if self._keep_open:
            return
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
        last_exc: Exception | None = None
        for _ in range(3):
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=2000)
            except PlaywrightTimeoutError:
                # Some pages stay "loading" due to background requests; still try to snapshot.
                pass

            try:
                return {
                    "url": self.page.url,
                    "title": self.page.title(),
                    "dom": self.page.evaluate(DOM_EXTRACTION_SCRIPT),
                }
            except PlaywrightError as exc:
                msg = str(exc)
                if "Execution context was destroyed" in msg or "Cannot find context" in msg:
                    last_exc = exc
                    self.page.wait_for_timeout(200)
                    continue
                raise

        raise RuntimeError("Failed to capture page state while page was navigating.") from last_exc

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
        locator = self.page.locator(selector)
        count = locator.count()
        if count == 0:
            raise ValueError(f"Selector not found on page: {selector}")

        chosen = None
        chosen_index = -1
        for idx in range(min(count, 30)):
            candidate = locator.nth(idx)
            try:
                if not candidate.is_visible():
                    continue
                if not candidate.is_enabled():
                    continue
                input_type = (candidate.get_attribute("type") or "").strip().lower()
                if input_type in {"checkbox", "radio", "hidden", "submit", "button"}:
                    continue
                chosen = candidate
                chosen_index = idx + 1
                break
            except PlaywrightError:
                continue

        if chosen is None:
            raise ValueError(
                f"No visible editable match for selector: {selector} (matched {count} elements)"
            )

        chosen.fill(text)
        return f"Typed {len(text)} chars into {selector} (match {chosen_index}/{count})"

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
    compact_dom: list[dict[str, Any]] = []
    for item in page_state.get("dom", [])[:MAX_DOM_ELEMENTS]:
        compact_dom.append(
            {
                "tag": item.get("tag"),
                "selector": item.get("selector"),
                "label": item.get("label"),
                "text": item.get("text"),
                "placeholder": item.get("placeholder"),
                "type": item.get("type"),
                "disabled": item.get("disabled", False),
            }
        )

    user_payload = {
        "task": task,
        "memory": memory.snapshot(),
        "page": {
            "url": page_state.get("url", ""),
            "title": page_state.get("title", ""),
            "interactive_elements": compact_dom,
        },
    }
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": json.dumps(user_payload)}],
        }
    )
    for attempt in range(1, MAX_CLAUDE_RETRIES + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=900,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages[-MAX_HISTORY_MESSAGES:],
            )
        except RateLimitError:
            if attempt == MAX_CLAUDE_RETRIES:
                raise
            # Back off to reduce token-per-minute pressure.
            time.sleep(2 ** attempt)

    raise RuntimeError("Claude request retry loop exited unexpectedly.")


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


def run_setup_mode() -> None:
    print("\n[setup mode] Browser is open. You can manually log in now.")
    print("Press Ctrl+C when finished, or this will wait indefinitely.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[setup mode] Login saved. Browser will close.")


def run_agent_loop(
    client: Anthropic,
    browser: BrowserAutomation,
    task: str,
    model: str,
    max_steps: int,
) -> None:
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


def run_agent(
    task: str,
    start_url: str | None,
    model: str,
    max_steps: int,
    headless: bool,
    timeout_ms: int,
    profile_dir: str | None = None,
    setup_mode: bool = False,
    use_personal_edge: bool = False,
    edge_user_data_dir: str | None = None,
    edge_profile_dir: str = "Default",
    keep_open: bool = False,
) -> None:
    load_dotenv_if_present()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not setup_mode and not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in environment variables or .env before running this script.")

    browser = BrowserAutomation(
        headless=headless,
        timeout_ms=timeout_ms,
        profile_dir=profile_dir,
        use_personal_edge=use_personal_edge,
        edge_user_data_dir=edge_user_data_dir,
        edge_profile_dir=edge_profile_dir,
        keep_open=keep_open,
    )

    try:
        if start_url:
            print(browser.navigate(start_url))

        if setup_mode:
            run_setup_mode()
            return

        client = Anthropic(api_key=api_key)
        run_agent_loop(client=client, browser=browser, task=task, model=model, max_steps=max_steps)
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
    parser.add_argument("--profile-dir", default="./my_profile", help="Browser profile directory for persistent login state")
    parser.add_argument("--setup-mode", action="store_true", help="Open browser for manual login (requires Ctrl+C to exit)")
    parser.add_argument("--use-personal-edge", action="store_true", help="Use your installed personal Microsoft Edge profile")
    parser.add_argument("--edge-user-data-dir", default=None, help="Edge User Data path (default: %LOCALAPPDATA%/Microsoft/Edge/User Data)")
    parser.add_argument("--edge-profile-dir", default="Default", help="Edge profile name like Default or Profile 1")
    parser.add_argument("--keep-open", action="store_true", help="Keep browser open when script finishes")
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
        use_personal_edge=args.use_personal_edge,
        edge_user_data_dir=args.edge_user_data_dir,
        edge_profile_dir=args.edge_profile_dir,
        keep_open=args.keep_open,
    )


if __name__ == "__main__":
    main()