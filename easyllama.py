#!/usr/bin/env python3
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
import ollama
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.segment import Segment
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import History
from prompt_toolkit.layout.processors import Processor, Transformation
from prompt_toolkit.layout.utils import explode_text_fragments

APP = "easyllama"
TAGLINE = "Enhanced Local LLM Shell"
DEFAULT_SYSTEM = ""
DEFAULT_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "5m")

def app_dir(xdg_var, fallback):
    """The XDG base directory for this app, e.g. ~/.config/easyllama (%APPDATA%\\easyllama on Windows)."""
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / APP
    return Path(os.environ.get(xdg_var) or Path.home() / fallback) / APP

CONFIG_DIR = app_dir("XDG_CONFIG_HOME", ".config")
DATA_DIR = app_dir("XDG_DATA_HOME", ".local/share")
STATE_DIR = app_dir("XDG_STATE_HOME", ".local/state")
SESSIONS_DIR = DATA_DIR / "sessions"
HISTORY_FILE = DATA_DIR / "history"
SETTINGS_FILE = CONFIG_DIR / "options.json"
SERVE_LOG = STATE_DIR / "serve.log"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN = (f"\033[{n}m" for n in range(31, 37))

def ok(s): print(f"{GREEN}{s}{RESET}")
def warn(s): print(f"{YELLOW}{s}{RESET}")
def err(s): print(f"{RED}{s}{RESET}")
def title(s): print(f"{BOLD}{CYAN}{s}{RESET}")

OFFLINE = (ConnectionError, httpx.TransportError)

server = None

def offline():
    if server and server.poll() is not None:
        code = server.returncode
        how = "stopped" if code == 0 else f"was killed by signal {-code}" if code < 0 \
            else f"exited with code {code}"
        err(f"The Ollama server started by /serve (pid {server.pid}) {how}. See {SERVE_LOG}")
    err(f"Ollama isn't running. Use {BOLD}/serve{RESET}{RED} to start it in the background.")

INPUT_STYLE = Style.from_dict({"": "#ffffff"})
menu_session = PromptSession(style=INPUT_STYLE)

def ask(prompt):
    return menu_session.prompt(ANSI(prompt)).strip()

COMMANDS = [
    ("/help", "show this help"),
    ("/clear", "clear the console"),
    ("/forget", "clear the chat session (start a new conversation)"),
    ("/undo", "remove the last exchange"),
    ("/retry", "regenerate the last reply"),
    ("/system [prompt]", "view or set the system prompt (/system none clears it)"),
    ("/set [option] [value]", "show or set a model option, e.g. /set temperature 0.7 (default resets)"),
    ("/context [size]", "show or set the context size, e.g. /context 4K, 16K, 64K"),
    ("/list", "list installed models (* = selected)"),
    ("/select [model]", "select a model by name or number from /list (0 = none)"),
    ("/<n>", "shortcut for /select <n>, e.g. /7 (/0 = none)"),
    ("/install [model]", "download a model, e.g. /install llama3.2"),
    ("/remove [model]", "delete an installed model"),
    ("/keep [on|off]", "keep the current model loaded when switching away (shown in /list)"),
    ("/load [model]", "load a model into memory without sending a message (default: the current one)"),
    ("/unload [model]", "free a loaded model from memory (default: the current one)"),
    ("/ps", "show loaded models (like ollama ps)"),
    ("/serve", "start the Ollama server in the background if it isn't running"),
    ("/shutdown", "stop the Ollama server started with /serve"),
    ("/info", "show model and session info"),
    ("/save", "save the session now"),
    ("/export [file]", "export the conversation as JSON (default: ./chat-<time>.json)"),
    ("/resume [n]", "resume a saved session"),
    ("/autosave [on|off]", "save the session after every reply"),
    ("/show-thinking [on|off]", "show the reasoning of thinking models"),
    ("/markdown [on|off]", "render replies as Markdown"),
    ("!<command>", "run a shell command, e.g. !ls -la or !cd src"),
    ("/exit", "quit"),
]
PARAMS = {
    "temperature": (float, "randomness, higher is more creative (e.g. 0.8)"),
    "top_p": (float, "only sample from the top tokens making up this probability (e.g. 0.9)"),
    "top_k": (int, "only sample from the k most likely tokens (e.g. 40)"),
    "min_p": (float, "drop tokens less likely than this fraction of the top one (e.g. 0.05)"),
    "seed": (int, "fixed seed for reproducible replies"),
    "num_predict": (int, "max tokens per reply (-1 = no limit)"),
    "repeat_penalty": (float, "penalize repeated tokens (e.g. 1.1)"),
    "repeat_last_n": (int, "how far back to look for repeats (e.g. 64)"),
    "presence_penalty": (float, "penalize tokens that already appeared"),
    "frequency_penalty": (float, "penalize tokens by how often they appeared"),
}

CONTEXT_SIZES = ["2K", "4K", "8K", "16K", "32K", "64K", "128K"]

def banner():
    print(f"{BOLD}{CYAN}{APP}{RESET} {DIM}· {TAGLINE}{RESET}")

def show_help():
    banner()
    title("Commands")
    for cmd, desc in COMMANDS:
        print(f"  {YELLOW}{cmd:<24}{RESET} {desc}")
    print(f"  {DIM}Tab completes commands and arguments, ↑/↓ browse history.{RESET}")
    print(f"  {DIM}From your shell, run one thing and exit: {APP} hello, {APP} /list, {APP} '!ls'{RESET}")

def parse_size(s):
    """'4K' -> 4096, '16k' -> 16384, '1M' -> 1048576, '8192' -> 8192."""
    s = s.strip().upper()
    mult = {"K": 1024, "M": 1024 * 1024}.get(s[-1:], 1)
    num = s[:-1] if mult > 1 else s
    return int(num) * mult if num.isdigit() and int(num) > 0 else None

def fmt_size(n):
    return f"{n // 1024}K" if n % 1024 == 0 else str(n)

def parse_toggle(args, current, name):
    if not args:
        state = f"{GREEN}on" if current else f"{RED}off"
        print(f"{name} is {state}{RESET}")
        return current
    if args[0].lower() in ("on", "off"):
        value = args[0].lower() == "on"
        ok(f"{name} {'on' if value else 'off'}.")
        return value
    warn(f"Usage: /{name.split()[0].lower()} [on|off]")
    return current

def installed_models():
    return list(dict.fromkeys(m.model for m in ollama.list().models))

def fmt_eta(secs):
    secs = int(secs)
    if secs >= 3600:
        return f"{secs // 3600}h{secs % 3600 // 60:02d}m"
    return f"{secs // 60}m{secs % 60:02d}s" if secs >= 60 else f"{secs}s"

def progress_line(status, done, total, speed):
    """  pulling 6a0746a1ec1a:  45% ━━━━━━━━━─────────── 1.2 GB/2.6 GB  38 MB/s  36s left"""
    pct = done * 100 // total
    info = f"{fmt_bytes(done)}/{fmt_bytes(total)}"
    if speed:
        info += f"  {fmt_bytes(speed)}/s"
        if done < total:
            info += f"  {fmt_eta((total - done) / speed)} left"
    head = f"  {status}: {pct:3d}% "
    width = min(30, shutil.get_terminal_size().columns - len(head) - len(info) - 2)
    bar = ""
    if width >= 10:
        filled = done * width // total
        bar = f"{GREEN}{'━' * filled}{DIM}{'─' * (width - filled)}{RESET} "
    return f"  {DIM}{status}:{RESET} {GREEN}{pct:3d}%{RESET} {bar}{info}"

def pull(name):
    print(f"{CYAN}Pulling {BOLD}{name}{RESET}{CYAN}...{RESET}")
    last = None
    samples = []
    try:
        for p in ollama.pull(name, stream=True):
            if p.status != last:
                print("\n" if last else "", end="")
                samples = []
                if not (p.total and p.completed):
                    print(f"  {DIM}{p.status}{RESET}", end="", flush=True)
            if p.total and p.completed:
                now = time.monotonic()
                samples = [s for s in samples if now - s[0] <= 3] + [(now, p.completed)]
                t0, c0 = samples[0]
                speed = (p.completed - c0) / (now - t0) if now - t0 >= 0.5 else None
                print(f"\r\033[K{progress_line(p.status, p.completed, p.total, speed)}", end="", flush=True)
            last = p.status
    finally:
        print()

def pick(models, choice):
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    return choice if choice in models else None

def item(key, label, selected=False):
    mark = f"  {GREEN}*{RESET}" if selected else ""
    color = f"{BOLD}{GREEN}" if selected else ""
    print(f"  {YELLOW}{key}){RESET} {color}{label}{RESET}{mark}")

def list_models(current=None, none=True, kept=()):
    models = installed_models()
    if none:
        item(0, "[none]", current is None)
    if not models:
        warn("  No models installed. Use /install [model].")
    for i, m in enumerate(models, 1):
        item(i, m + (f"  {DIM}(keep){RESET}" if m in kept else ""), m == current)
    return models

def menu(heading, current=None, none=False):
    while True:
        title(heading)
        models = list_models(current, none)
        if not models and not none:
            return None
        item("b", "back")
        choice = ask(f"{CYAN}>{RESET} ")
        if choice.lower() == "b":
            return None
        if none and choice == "0":
            return ""
        if name := pick(models, choice):
            return name
        err("Invalid choice.")

def install(name):
    if not name:
        name = ask(f"Model name to install (e.g. llama3.2), {YELLOW}b){RESET} back: ")
        if not name or name.lower() == "b":
            return None
    try:
        pull(name)
        ok(f"Installed {name}. Use /select {name} to use it.")
        return name
    except ollama.ResponseError as e:
        if "file does not exist" in e.error:
            err(f"Install failed: there is no model called {name}. "
                f"Check the name at https://ollama.com/library")
        else:
            err(f"Install failed: {e.error}")

def remove(choice, current):
    name = pick(installed_models(), choice) if choice else menu("Remove which model?", current)
    if choice and not name:
        err(f"Model {choice} is not installed. See /list.")
    if not name or ask(f"{RED}Really remove {BOLD}{name}{RESET}{RED}?{RESET} [y/N] ").lower() != "y":
        return None
    try:
        ollama.delete(name)
        ok(f"Removed {name}.")
        return name
    except ollama.ResponseError as e:
        err(f"Remove failed: {e.error}")

def select(choice, current):
    """Returns the chosen model, "" for none, or None if nothing changed."""
    if choice in ("0", "none", "[none]"):
        name = ""
    elif choice:
        name = pick(installed_models(), choice)
        if not name:
            if choice.isdigit():
                err(f"There is no model {choice}. See /list.")
            else:
                err(f"Model {choice} is not installed. See /list or /install {choice}.")
    else:
        name = menu("Select a model", current, none=True)
    if name:
        ok(f"Using {name}.")
    elif name == "":
        warn("No model selected.")
    return name

class State:
    def __init__(self):
        self.model = None
        self.num_ctx = 65536
        self.autosave = True
        self.show_thinking = True
        self.markdown = True
        self.system_prompt = DEFAULT_SYSTEM
        self.history_length = 1000
        self.keep = []
        self.params = {}
        self.load_settings()
        self.new_session()

    def load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text())
        except OSError:
            self.save_settings()
            return
        except ValueError:
            warn(f"{SETTINGS_FILE} is not valid JSON; using default options.")
            return
        self.num_ctx = parse_size(str(data.get("context", data.get("num_ctx", "")))) or self.num_ctx
        self.autosave = data.get("autosave", self.autosave)
        self.show_thinking = data.get("show_thinking", self.show_thinking)
        self.markdown = data.get("markdown", self.markdown)
        params = data.get("options")
        if isinstance(params, dict):
            self.params = {k: v for k, v in params.items() if k in PARAMS}
        keep = data.get("keep_loaded", data.get("keep"))
        if isinstance(keep, list):
            self.keep = keep
        self.system_prompt = data.get("system_prompt") or ""
        length = data.get("history_length")
        if isinstance(length, int) and length > 0:
            self.history_length = length
        last = data.get("last_model")
        if last:
            try:
                if last in installed_models():
                    self.model = last
                else:
                    warn(f"Last model {last} is no longer installed.")
            except OFFLINE:
                self.model = last
            except Exception:
                warn("Could not reach Ollama to restore the last model.")
        missing = {k: v for k, v in self.settings().items() if k not in data}
        old_keys = ("num_ctx", "keep_models", "keep")
        if missing or any(k in data for k in old_keys):
            for k in old_keys:
                data.pop(k, None)
            data.update(missing)
            SETTINGS_FILE.write_text(json.dumps(data, indent=2))

    def settings(self):
        return {
            "context": fmt_size(self.num_ctx),
            "autosave": self.autosave,
            "show_thinking": self.show_thinking,
            "markdown": self.markdown,
            "keep_loaded": self.keep,
            "last_model": self.model,
            "system_prompt": self.system_prompt or None,
            "history_length": self.history_length,
            "options": self.params,
        }

    def save_settings(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(self.settings(), indent=2))

    def release(self):
        """Unload the current model before switching away, unless /keep is on for it."""
        if self.model not in self.keep:
            unload(self.model)

    def new_session(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.used = 0
        self.stats = None
        self.think_time = None
        self.path = None

    def save(self, quiet=False):
        if len(self.messages) < 2 and not self.path:
            if not quiet:
                warn("Nothing to save yet.")
            return
        if not self.path:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            self.path = SESSIONS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        data = {"model": self.model, "num_ctx": self.num_ctx, "messages": self.messages}
        self.path.write_text(json.dumps(data, indent=2))
        if not quiet:
            ok(f"Saved session to {self.path}")

    def changed(self):
        if self.autosave:
            self.save(quiet=True)

def unload(model):
    """Frees a model from memory (like `ollama stop`). Returns True if it was loaded."""
    if not model or model not in loaded_models():
        return False
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
        return True
    except Exception:
        return False

def set_keep_alive(model, keep):
    """If the model is loaded, switch it between never timing out and Ollama's default timeout."""
    try:
        if any(m.model == model for m in ollama.ps().models):
            ollama.generate(model=model, prompt="", keep_alive=-1 if keep else DEFAULT_KEEP_ALIVE)
    except Exception:
        pass

def saved_sessions():
    return sorted(SESSIONS_DIR.glob("*.json"), reverse=True) if SESSIONS_DIR.exists() else []

def session_preview(path):
    data = json.loads(path.read_text())
    first = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
    first = " ".join(first.split())
    first = first[:50] + "…" if len(first) > 50 else first
    when = time.strftime("%Y-%m-%d %H:%M", time.strptime(path.stem, "%Y%m%d-%H%M%S"))
    count = len(data["messages"]) - 1
    return f"{when}  {CYAN}{data.get('model') or '[none]'}{RESET}  {DIM}{count} msgs{RESET}  {first}"

def resume(st, choice):
    sessions = saved_sessions()[:20]
    if not sessions:
        warn("No saved sessions.")
        return
    if not choice:
        title("Resume which session?")
        for i, p in enumerate(sessions, 1):
            item(i, session_preview(p), p == st.path)
        item("b", "back")
        choice = ask(f"{CYAN}>{RESET} ")
        if choice.lower() == "b":
            return
    if not (choice.isdigit() and 1 <= int(choice) <= len(sessions)):
        err("Invalid choice.")
        return
    path = sessions[int(choice) - 1]
    data = json.loads(path.read_text())
    st.messages, st.path, st.used = data["messages"], path, 0
    st.num_ctx = data.get("num_ctx", st.num_ctx)
    model = data.get("model")
    if model and model in installed_models():
        if model != st.model:
            st.release()
        st.model = model
        st.save_settings()
    elif model:
        warn(f"Model {model} from that session is not installed; keeping {st.model or '[none]'}.")
    ok(f"Resumed session with {len(st.messages) - 1} messages.")
    for m in st.messages[1:][-4:]:
        who = f"{BLUE}you" if m["role"] == "user" else f"{GREEN}ai"
        text = " ".join(m["content"].split())
        print(f"  {BOLD}{who}{RESET}: {text[:100] + '…' if len(text) > 100 else text}")

class Spinner:
    """Shows an animated "label   " / "label.  " / "label.. " / "label..." on the current line."""
    def __init__(self, label):
        self.label = label
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        frames = ["   ", ".  ", ".. ", "..."]
        i = 0
        while not self.stop_event.is_set():
            print(f"\r\033[K{DIM}{MAGENTA}{self.label}{frames[i % len(frames)]}{RESET}", end="", flush=True)
            i += 1
            self.stop_event.wait(0.4)

    def stop(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.thread.join()
            print("\r\033[K", end="", flush=True)

class Block:
    """Renders a Markdown block without rich's leading blank line (lists get one).
    With tail=True only the last screenful is shown, so a live block taller than the terminal doesn't break."""
    def __init__(self, text, tail=False):
        self.text, self.tail = text, tail

    def __rich_console__(self, console, options):
        lines = console.render_lines(Markdown(self.text), options, pad=False)
        while lines and not any(seg.text.strip() or (seg.style and seg.style.bgcolor) for seg in lines[0]):
            lines.pop(0)
        if self.tail:
            lines = lines[-max(console.size.height - 1, 1):]
        for line in lines:
            yield from line
            yield Segment.line()

class MarkdownStream:
    """Streams Markdown: finished blocks are printed for good, the block being written is shown live."""
    def __init__(self):
        self.console = Console()
        self.pending = ""
        self.live = None

    def split_point(self):
        pos, cut, fence = 0, 0, False
        for line in self.pending.splitlines(keepends=True):
            pos += len(line)
            if line.lstrip().startswith(("```", "~~~")):
                fence = not fence
            elif not fence and not line.strip() and line.endswith("\n"):
                cut = pos
        return cut

    def feed(self, text):
        self.pending += text
        if cut := self.split_point():
            done, self.pending = self.pending[:cut], self.pending[cut:]
            if done.strip():
                self.console.print(Block(done))
                self.console.print()
        if self.pending.strip():
            if not self.live:
                self.live = Live(console=self.console, transient=True, auto_refresh=False)
                self.live.start()
            self.live.update(Block(self.pending, tail=True), refresh=True)

    def finish(self):
        if self.live:
            self.live.stop()
            self.live = None
        if self.pending.strip():
            self.console.print(Block(self.pending))
        self.pending = ""

def stream_reply(st):
    reply = ""
    thinking = False
    think_start = think_end = None
    messages = [m for m in st.messages if m["role"] != "system" or m["content"]]
    spinner = Spinner("Loading")
    md = MarkdownStream() if st.markdown else None
    try:
        for chunk in ollama.chat(model=st.model, messages=messages, stream=True,
                                 options={**st.params, "num_ctx": st.num_ctx},
                                 keep_alive=-1 if st.model in st.keep else DEFAULT_KEEP_ALIVE):
            msg = chunk.message
            if msg.thinking:
                if not thinking:
                    thinking = True
                    think_start = time.monotonic()
                    if st.show_thinking:
                        spinner.stop()
                        print(f"{DIM}{MAGENTA}Thinking...")
                    else:
                        spinner.label = "Thinking"
                if st.show_thinking:
                    print(msg.thinking, end="", flush=True)
            if msg.content:
                spinner.stop()
                if thinking:
                    think_end = time.monotonic()
                    if st.show_thinking:
                        print(f"\n...done thinking.{RESET}\n")
                    thinking = False
                if md:
                    md.feed(msg.content)
                else:
                    print(msg.content, end="", flush=True)
                reply += msg.content
    finally:
        spinner.stop()
        if md:
            md.finish()
        print(RESET, end="")
    if not md:
        print()
    st.used = (chunk.prompt_eval_count or 0) + (chunk.eval_count or 0)
    st.stats = chunk
    st.think_time = (think_end or time.monotonic()) - think_start if think_start else None
    return reply

def loaded_models():
    try:
        return {m.model for m in ollama.ps().models}
    except Exception:
        return set()

def warn_evicted(st, before, model=None):
    """Ollama unloads other models when the new one needs the GPU memory, even /keep ones."""
    model = model or st.model
    for name in sorted((before - loaded_models()) & set(st.keep)):
        if name != model:
            warn(f"Ollama unloaded {name} (keep) to make room for {model} on the GPU.")

def load_model(st, model=None):
    """Loads a model (default: the current one) with the session's context size,
    so the first message doesn't wait for it."""
    model = model or st.model
    loaded = next((m for m in ollama.ps().models if m.model == model), None)
    if loaded and loaded.context_length == st.num_ctx:
        warn(f"{model} is already loaded. See /ps.")
        return
    before = loaded_models()
    spinner = Spinner("Loading")
    start = time.monotonic()
    try:
        ollama.generate(model=model, prompt="", options={"num_ctx": st.num_ctx},
                        keep_alive=-1 if model in st.keep else DEFAULT_KEEP_ALIVE)
    except KeyboardInterrupt:
        spinner.stop()
        warn("[interrupted]")
        return
    except ollama.ResponseError as e:
        spinner.stop()
        err(f"Error: {e.error}")
        return
    except OFFLINE:
        spinner.stop()
        offline()
        return
    spinner.stop()
    ok(f"Loaded {model} ({fmt_size(st.num_ctx)} context) in {time.monotonic() - start:.1f}s")
    warn_evicted(st, before, model)

def generate(st):
    """Answer the last user message; on failure the user message is dropped."""
    before = loaded_models()
    try:
        reply = stream_reply(st)
    except KeyboardInterrupt:
        print(RESET)
        warn("[interrupted]")
        st.messages.pop()
        warn_evicted(st, before)
        return
    except ollama.ResponseError as e:
        print(RESET)
        err(f"Error: {e.error}")
        st.messages.pop()
        warn_evicted(st, before)
        return
    except OFFLINE:
        print(RESET)
        offline()
        st.messages.pop()
        return
    st.messages.append({"role": "assistant", "content": reply})
    st.changed()
    warn_evicted(st, before)

def set_param(st, args):
    if not args:
        title("Options")
        for name, (_, desc) in PARAMS.items():
            value = str(st.params.get(name, "default")).ljust(10)
            color = CYAN if name in st.params else DIM
            print(f"  {YELLOW}{name:<18}{RESET} {color}{value}{RESET} {DIM}{desc}{RESET}")
        print(f"  {DIM}/set <option> <value> to change, /set <option> default to reset.{RESET}")
        return
    name = args[0].lower()
    if name not in PARAMS:
        err(f"Unknown option {name}. Options: {', '.join(PARAMS)}")
        return
    if len(args) < 2:
        print(f"{name} is {st.params.get(name, 'default')}")
        return
    if args[1].lower() in ("default", "none", "reset"):
        st.params.pop(name, None)
        st.save_settings()
        ok(f"{name} reset to the model's default.")
        return
    kind = PARAMS[name][0]
    try:
        st.params[name] = kind(args[1])
    except ValueError:
        err(f"{name} needs {'a whole number' if kind is int else 'a number'}, e.g. /set {name} "
            f"{'42' if kind is int else '0.7'}")
        return
    st.save_settings()
    ok(f"{name} set to {st.params[name]}.")

def export(st, target):
    if len(st.messages) < 2:
        warn("Nothing to export yet.")
        return
    name = f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path = Path(os.path.expandvars(os.path.expanduser(target))) if target else Path(name)
    if path.is_dir():
        path = path / name
    data = {
        "model": st.model,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "context": st.num_ctx,
        "options": st.params,
        "messages": [m for m in st.messages if m["role"] != "system" or m["content"]],
    }
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    except OSError as e:
        err(f"Export failed: {e.strerror}: {path}")
        return
    ok(f"Exported {len(data['messages'])} messages to {path.resolve()}")

def show_info(st):
    try:
        s = ollama.show(st.model)
    except ollama.ResponseError as e:
        err(f"Info failed: {e.error}")
        return
    d, mi = s.details, s.modelinfo or {}
    arch = mi.get("general.architecture", d.family)
    max_ctx = mi.get(f"{arch}.context_length")
    def row(k, v): print(f"  {DIM}{k:<15}{RESET} {v}")
    title("Model")
    row("name", f"{BOLD}{GREEN}{st.model}{RESET}")
    row("architecture", arch)
    row("parameters", d.parameter_size)
    row("quantization", d.quantization_level)
    if max_ctx:
        row("max context", f"{fmt_size(max_ctx)} ({max_ctx} tokens)")
    if s.capabilities:
        row("capabilities", ", ".join(s.capabilities))
    show_loaded(st, row)
    pct = st.used * 100 // st.num_ctx
    color = RED if pct >= 90 else YELLOW if pct >= 70 else GREEN
    title("Session")
    capped = f"  {YELLOW}(model max is {fmt_size(max_ctx)}){RESET}" if max_ctx and st.num_ctx > max_ctx else ""
    row("context size", f"{fmt_size(st.num_ctx)} ({st.num_ctx} tokens){capped}")
    row("context used", f"{color}{st.used} tokens ({pct}%){RESET}")
    row("saved to", st.path or "not saved")
    if st.params:
        row("options", ", ".join(f"{k}={v}" for k, v in st.params.items()))
    show_stats(st, row)

def fmt_bytes(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"

def show_loaded(st, row):
    """Size and processor, as shown by `ollama ps`."""
    try:
        m = next((m for m in ollama.ps().models if m.model == st.model), None)
    except Exception:
        return
    if not m:
        row("size", f"{DIM}not loaded{RESET}")
        row("processor", f"{DIM}not loaded{RESET}")
        return
    text, color = processor(m)
    row("size", fmt_bytes(m.size))
    row("processor", f"{color}{text}{RESET}")

def processor(m):
    """The PROCESSOR column of `ollama ps` and its color."""
    gpu = round(m.size_vram * 100 / m.size) if m.size else 0
    if gpu >= 100:
        return "100% GPU", GREEN
    if gpu <= 0:
        return "100% CPU", YELLOW
    return f"{100 - gpu}%/{gpu}% CPU/GPU", YELLOW

def until(expires_at):
    """The UNTIL column of `ollama ps`."""
    secs = (expires_at - datetime.now(timezone.utc)).total_seconds()
    if secs > 100 * 365 * 86400:
        return "Forever"
    if secs <= 0:
        return "Stopping..."
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if secs >= size:
            n = int(secs // size)
            return f"{n} {unit}{'s' if n > 1 else ''} from now"
    return "Less than a minute from now"

def show_ps():
    """Loaded models, like `ollama ps`."""
    models = ollama.ps().models
    if not models:
        print(f"{DIM}No models loaded.{RESET}")
        return
    rows = [(m.model, fmt_bytes(m.size), *processor(m), str(m.context_length or ""), until(m.expires_at))
            for m in models]
    heads = ("NAME", "SIZE", "PROCESSOR", "CONTEXT", "UNTIL")
    widths = [max(len(h), *(len(r[i + (i > 2)]) for r in rows)) for i, h in enumerate(heads)]
    print(BOLD + "    ".join(h.ljust(w) for h, w in zip(heads, widths)).rstrip() + RESET)
    for name, size, proc, color, ctx, left in rows:
        cells = [name.ljust(widths[0]), size.ljust(widths[1]),
                 f"{color}{proc.ljust(widths[2])}{RESET}", ctx.ljust(widths[3]), left]
        print("    ".join(cells))

def ollama_running():
    try:
        ollama.ps()
        return True
    except OFFLINE:
        return False

def serve():
    """Starts `ollama serve` detached from this process, so it keeps running after /exit."""
    if ollama_running():
        ok("Ollama is already running.")
        return
    exe = shutil.which("ollama")
    if not exe:
        err("The ollama command was not found. Install it from https://ollama.com/download")
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with SERVE_LOG.open("ab") as log:
        proc = subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    global server
    server = proc
    spinner = Spinner("Starting Ollama")
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                spinner.stop()
                err(f"ollama serve exited with code {proc.returncode}. See {SERVE_LOG}")
                return
            if ollama_running():
                spinner.stop()
                ok(f"Ollama is running in the background (pid {proc.pid}, log: {SERVE_LOG}).")
                return
            time.sleep(0.25)
    finally:
        spinner.stop()
    warn(f"Ollama was started (pid {proc.pid}) but isn't answering yet. See {SERVE_LOG}")

def serve_pids():
    """Your own `ollama serve` processes: from this /serve, an earlier one, or a terminal."""
    pids = []
    try:
        procs = [d for d in Path("/proc").iterdir() if d.name.isdigit()]
    except OSError:
        procs = []
    for d in procs:
        try:
            if d.stat().st_uid != os.getuid():
                continue
            args = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(args) > 1 and os.path.basename(args[0]) == b"ollama" and args[1] == b"serve":
            pids.append(int(d.name))
    if server and server.poll() is None and server.pid not in pids:
        pids.append(server.pid)
    return pids

def shutdown():
    global server
    if not ollama_running():
        server = None
        warn("Ollama isn't running.")
        return
    pids = serve_pids()
    if not pids:
        warn("Ollama wasn't started by you, so it can't be stopped from here. "
             "If it's the system service: sudo systemctl stop ollama")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    spinner = Spinner("Stopping Ollama")
    try:
        deadline = time.monotonic() + 10
        while ollama_running() and time.monotonic() < deadline:
            time.sleep(0.25)
    finally:
        spinner.stop()
    if server:
        server.poll()
        server = None
    if ollama_running():
        warn(f"Ollama is still running (pid {', '.join(map(str, pids))}).")
    else:
        ok("Ollama stopped.")

def fmt_secs(ns):
    s = ns / 1e9
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s"

def show_stats(st, row):
    s = st.stats
    if not s or not s.eval_count:
        return
    title("Speed (last reply)")
    if s.eval_duration:
        row("speed", f"{BOLD}{s.eval_count / (s.eval_duration / 1e9):.1f} tokens/s{RESET}")
    think = (st.think_time or 0) * 1e9
    row("load time", fmt_secs(s.load_duration or 0))
    row("prompt time", fmt_secs(s.prompt_eval_duration or 0))
    row("thinking time", fmt_secs(think))
    row("answer time", fmt_secs(max((s.eval_duration or 0) - think, 0)))
    if s.total_duration:
        row("total time", fmt_secs(s.total_duration))

def run_shell(command):
    if not command:
        warn("Usage: !<command>, e.g. !ls -la")
        return
    if command == "cd" or command.startswith("cd "):
        target = os.path.expandvars(os.path.expanduser(command[2:].strip() or "~"))
        try:
            os.chdir(target)
            print(f"{DIM}{os.getcwd()}{RESET}")
        except OSError as e:
            err(f"cd: {e.strerror}: {target}")
        return
    try:
        code = subprocess.run([os.environ.get("SHELL", "/bin/sh"), "-c", command]).returncode
    except KeyboardInterrupt:
        code = 130
        print()
    if code and code != 130:
        err(f"[exit {code}]")

def path_options(text):
    expanded = os.path.expanduser(text)
    folder, prefix = os.path.split(expanded)
    try:
        names = os.listdir(folder or ".")
    except OSError:
        return []
    out = []
    for n in sorted(names):
        if n.startswith(prefix) and (prefix.startswith(".") or not n.startswith(".")):
            full = os.path.join(folder, n)
            shown = text[:len(text) - len(prefix)] + n
            out.append(shown + "/" if os.path.isdir(full) else shown + " ")
    return out

def executables(prefix):
    found = set()
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            found.update(n for n in os.listdir(d) if n.startswith(prefix)
                         and os.access(os.path.join(d, n), os.X_OK))
        except OSError:
            pass
    return sorted(found)

def shell_matches(words, text):
    if len(words) <= 1:
        word = text[1:]
        if "/" in word:
            return ["!" + o for o in path_options(word)]
        return ["!" + o + " " for o in executables(word)]
    return path_options(text)

class ShellCompleter(Completer):
    def get_completions(self, document, complete_event):
        line = document.text_before_cursor
        words = line.split()
        if line.endswith(" ") or not words:
            words.append("")
        text = words[-1]
        if line.startswith("!"):
            matches = shell_matches(words, text)
        elif not line.startswith("/"):
            return
        elif len(words) <= 1:
            matches = [c.split()[0] + " " for c, _ in COMMANDS if c.startswith(text)]
        elif words[0] == "/export":
            matches = path_options(text)
        else:
            cmd = words[0]
            if cmd == "/set":
                options = list(PARAMS) if len(words) == 2 else ["default"]
            elif cmd == "/unload":
                options = sorted(loaded_models())
            elif cmd == "/load":
                try:
                    options = [m for m in installed_models() if m not in loaded_models()]
                except Exception:
                    options = []
            elif cmd in ("/select", "/remove"):
                try:
                    options = installed_models() + (["[none]"] if cmd == "/select" else [])
                except Exception:
                    options = []
            elif cmd in ("/autosave", "/show-thinking", "/markdown", "/keep"):
                options = ["on", "off"]
            elif cmd == "/context":
                options = CONTEXT_SIZES
            elif cmd == "/resume":
                options = [str(i) for i in range(1, len(saved_sessions()[:20]) + 1)]
            else:
                options = []
            matches = [o + " " for o in options if o.startswith(text)]
        for m in matches:
            yield Completion(m, start_position=-len(text), display=m.rstrip())

class LineHistory(History):
    """Plain history file, one entry per line (like readline's)."""
    def __init__(self, path, limit=1000):
        super().__init__()
        self.path, self.limit = path, limit

    def load_history_strings(self):
        try:
            lines = [l for l in self.path.read_text().splitlines() if l]
        except OSError:
            return []
        if len(lines) > self.limit:
            lines = lines[-self.limit:]
            self.path.write_text("".join(l + "\n" for l in lines))
        return list(reversed(lines))

    def store_string(self, string):
        with self.path.open("a") as f:
            f.write(string.replace("\n", " ") + "\n")

class HidePrefix(Processor):
    """Hides a leading / or ! in the input; the prompt shows it in place of the $."""
    def apply_transformation(self, ti):
        if ti.lineno == 0 and ti.document.text[:1] in ("/", "!"):
            fragments = explode_text_fragments(ti.fragments)[1:]
            return Transformation(fragments,
                                  source_to_display=lambda i: max(i - 1, 0),
                                  display_to_source=lambda i: i + 1)
        return Transformation(ti.fragments)

PROMPT_CHARS = {"/": YELLOW, "!": RED}

def make_session(st):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=LineHistory(HISTORY_FILE, st.history_length),
        completer=ShellCompleter(),
        complete_in_thread=True,
        complete_while_typing=False,
        complete_style=CompleteStyle.READLINE_LIKE,
        input_processors=[HidePrefix()],
        style=INPUT_STYLE,
    )
    buf = session.default_buffer

    def keep_cursor_after_prefix(_):
        if buf.text[:1] in PROMPT_CHARS and buf.cursor_position == 0:
            buf.cursor_position = 1
    buf.on_cursor_position_changed += keep_cursor_after_prefix
    buf.on_text_changed += keep_cursor_after_prefix

    def message():
        label = f"{GREEN}{st.model}" if st.model else f"{DIM}[none]"
        char = buf.text[:1] if buf.text[:1] in PROMPT_CHARS else "$"
        return ANSI(f"{BOLD}{label}{RESET} {PROMPT_CHARS.get(char, BLUE)}{BOLD}{char}{RESET} ")
    return session, message

def command(st, user):
    """Runs a /command. Returns "exit" for /exit, False for an unknown command, True otherwise."""
    cmd, *args = user.split()
    rest = user[len(cmd):].strip()
    if cmd == "/exit":
        return "exit"
    elif cmd == "/help":
        show_help()
    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
    elif cmd == "/forget":
        st.new_session()
        ok("Session cleared.")
    elif cmd == "/undo":
        if st.messages[-1]["role"] == "assistant":
            st.messages.pop()
        if st.messages[-1]["role"] == "user":
            st.messages.pop()
            st.changed()
            ok("Removed the last exchange.")
        else:
            warn("Nothing to undo.")
    elif cmd == "/retry":
        if not st.model:
            warn("No model selected. Use /select [model].")
        elif len(st.messages) < 2:
            warn("Nothing to retry.")
        else:
            if st.messages[-1]["role"] == "assistant":
                st.messages.pop()
            generate(st)
    elif cmd == "/system":
        if rest:
            st.system_prompt = "" if rest.lower() in ("none", "[none]") else rest
            st.messages[0] = {"role": "system", "content": st.system_prompt}
            st.changed()
            st.save_settings()
            ok("System prompt updated." if st.system_prompt else "System prompt cleared.")
        else:
            title("System prompt")
            print(st.messages[0]["content"] or f"{DIM}[none]{RESET}")
    elif cmd == "/list":
        list_models(st.model, kept=st.keep)
    elif cmd == "/install":
        install(rest)
    elif cmd == "/remove":
        if remove(rest, st.model) == st.model:
            st.model = None
            st.save_settings()
    elif cmd == "/select" or cmd[1:].isdigit():
        name = select(rest if cmd == "/select" else cmd[1:], st.model)
        if name is not None:
            if (name or None) != st.model:
                st.release()
            st.model, st.used, st.stats = name or None, 0, None
            st.save_settings()
    elif cmd == "/load":
        name = pick(installed_models(), rest) if rest else st.model
        if not rest and not name:
            warn("No model selected. Use /load [model].")
        elif not name:
            err(f"Model {rest} is not installed. See /list or /install {rest}.")
        else:
            load_model(st, name)
    elif cmd == "/unload":
        name = rest or st.model
        if not name:
            warn("No model selected. Use /unload [model].")
        elif not ollama_running():
            offline()
        elif unload(name):
            ok(f"Unloaded {name}.")
        else:
            warn(f"{name} is not loaded. See /ps.")
    elif cmd == "/ps":
        show_ps()
    elif cmd == "/serve":
        serve()
    elif cmd == "/shutdown":
        shutdown()
    elif cmd == "/info":
        if st.model:
            show_info(st)
        else:
            warn("No model selected. Use /select [model].")
    elif cmd == "/context":
        size = parse_size(args[0]) if args else None
        if not args:
            print(f"Context size: {CYAN}{fmt_size(st.num_ctx)}{RESET} ({st.num_ctx} tokens)")
        elif size:
            st.num_ctx = size
            st.save_settings()
            ok(f"Context size set to {fmt_size(size)} ({size} tokens)")
        else:
            warn("Usage: /context [size], e.g. /context 4K, 16K, 128K, 8192")
    elif cmd == "/set":
        set_param(st, args)
    elif cmd == "/export":
        export(st, rest)
    elif cmd == "/save":
        st.save()
    elif cmd == "/resume":
        resume(st, rest)
    elif cmd == "/autosave":
        st.autosave = parse_toggle(args, st.autosave, "Autosave")
        st.save_settings()
        st.changed()
    elif cmd == "/show-thinking":
        st.show_thinking = parse_toggle(args, st.show_thinking, "Show-thinking")
        st.save_settings()
    elif cmd == "/markdown":
        st.markdown = parse_toggle(args, st.markdown, "Markdown")
        st.save_settings()
    elif cmd == "/keep":
        if not st.model:
            warn("No model selected. Use /select [model] first.")
        else:
            on = parse_toggle(args, st.model in st.keep, f"Keep for {st.model}")
            if on and st.model not in st.keep:
                st.keep.append(st.model)
            elif not on and st.model in st.keep:
                st.keep.remove(st.model)
            st.save_settings()
            if args:
                set_keep_alive(st.model, on)
    else:
        err(f"Unknown command {cmd}. Type /help.")
        return False
    return True

def run_once(args):
    """`easyllama hello` asks the current model, `easyllama /list` runs a command,
    `easyllama !ls` a shell command, then it exits."""
    st = State()
    line = " ".join(args).strip()
    if line in ("-h", "--help"):
        line = "/help"
    try:
        if line.startswith("!"):
            run_shell(line[1:].strip())
            return 0
        if line.startswith("/"):
            return 0 if command(st, line) else 1
        if not st.model:
            warn("No model selected. Use /select [model].")
            return 1
        st.autosave = False
        st.messages.append({"role": "user", "content": line})
        generate(st)
        return 0 if st.messages[-1]["role"] == "assistant" else 1
    except KeyboardInterrupt:
        print(RESET)
        return 130
    except OFFLINE:
        offline()
        return 1

def main():
    if len(sys.argv) > 1:
        sys.exit(run_once(sys.argv[1:]))
    st = State()
    session, message = make_session(st)
    banner()
    if st.model:
        ok(f"Using {st.model}. Type /help for commands.")
    else:
        warn(f"No model selected. Type {BOLD}/help{RESET}{YELLOW} for commands.")
    if not ollama_running():
        offline()

    while True:
        try:
            user = session.prompt(message).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        if not user:
            continue

        try:
            if user.startswith("!"):
                run_shell(user[1:].strip())
                continue

            if user.startswith("/"):
                if command(st, user) == "exit":
                    break
                continue

            if not st.model:
                warn("No model selected. Use /list and /select [model].")
                continue
            st.messages.append({"role": "user", "content": user})
            generate(st)
        except (KeyboardInterrupt, EOFError):
            print(RESET)
        except OFFLINE:
            offline()
    st.release()

if __name__ == "__main__":
    main()
