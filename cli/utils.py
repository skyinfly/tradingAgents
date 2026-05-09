import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.models import AnalystType
from tradingagents.llm_clients.model_catalog import get_model_options

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Media Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def get_ticker() -> str:
    """提示用户输入股票代码。"""
    ticker = questionary.text(
        f"请输入要分析的准确股票代码（{TICKER_INPUT_EXAMPLES}）：",
        validate=lambda x: len(x.strip()) > 0 or "请输入有效的股票代码。",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print("\n[red]未提供股票代码，正在退出...[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """规范化股票代码输入，同时保留交易所后缀。"""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """提示用户输入 YYYY-MM-DD 格式的日期。"""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        "请输入分析日期（YYYY-MM-DD）：",
        validate=lambda x: validate_date(x.strip())
        or "请输入有效的 YYYY-MM-DD 格式日期。",
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print("\n[red]未提供日期，正在退出...[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """通过交互式复选框选择分析师。"""
    choices = questionary.checkbox(
        "请选择你的【分析师团队】：",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction="\n- 按空格键选择/取消选择分析师\n- 按 'a' 键全选/取消全选\n- 完成后按回车",
        validate=lambda x: len(x) > 0 or "至少需要选择一名分析师。",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print("\n[red]未选择任何分析师，正在退出...[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """通过交互式选择设置研究深度。"""

    # 定义研究深度选项及其对应数值
    DEPTH_OPTIONS = [
        ("浅度 - 快速研究，较少的辩论和策略讨论轮次", 1),
        ("中度 - 中间档，适度的辩论和策略讨论轮次", 3),
        ("深度 - 全面研究，深入的辩论和策略讨论", 5),
    ]

    choice = questionary.select(
        "请选择你的【研究深度】：",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction="\n- 使用方向键导航\n- 按回车选择",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]未选择研究深度，正在退出...[/red]")
        exit(1)

    return choice


def _fetch_openrouter_models() -> List[Tuple[str, str]]:
    """从 OpenRouter API 获取可用模型。"""
    import requests
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        return [(m.get("name") or m["id"], m["id"]) for m in models]
    except Exception as e:
        console.print(f"\n[yellow]无法获取 OpenRouter 模型：{e}[/yellow]")
        return []


def select_openrouter_model() -> str:
    """从最新可用模型中选择一个 OpenRouter 模型，或输入自定义 ID。"""
    models = _fetch_openrouter_models()

    choices = [questionary.Choice(name, value=mid) for name, mid in models[:5]]
    choices.append(questionary.Choice("Custom model ID", value="custom"))

    choice = questionary.select(
        "选择 OpenRouter 模型（最新可用）：",
        choices=choices,
        instruction="\n- 使用方向键导航\n- 按回车选择",
        style=questionary.Style([
            ("selected", "fg:magenta noinherit"),
            ("highlighted", "fg:magenta noinherit"),
            ("pointer", "fg:magenta noinherit"),
        ]),
    ).ask()

    if choice is None or choice == "custom":
        return questionary.text(
            "请输入 OpenRouter 模型 ID（例如 google/gemma-4-26b-a4b-it）：",
            validate=lambda x: len(x.strip()) > 0 or "请输入模型 ID。",
        ).ask().strip()

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """通过交互式选择设置浅思考 LLM 引擎。"""

    if provider.lower() == "openrouter":
        return select_openrouter_model()

    choice = questionary.select(
        "请选择你的【快速思考 LLM 引擎】：",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, "quick")
        ],
        instruction="\n- 使用方向键导航\n- 按回车选择",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(
            "\n[red]未选择浅思考 LLM 引擎，正在退出...[/red]"
        )
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """通过交互式选择设置深思考 LLM 引擎。"""

    if provider.lower() == "openrouter":
        return select_openrouter_model()

    choice = questionary.select(
        "请选择你的【深度思考 LLM 引擎】：",
        choices=[
            questionary.Choice(display, value=value)
            for display, value in get_model_options(provider, "deep")
        ],
        instruction="\n- 使用方向键导航\n- 按回车选择",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print("\n[red]未选择深思考 LLM 引擎，正在退出...[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str | None]:
    """选择 LLM 提供商及其 API 端点。"""
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Google", None),  # google-genai SDK manages its own endpoint
        ("Anthropic", "https://api.anthropic.com/"),
        ("xAI", "https://api.x.ai/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),
    ]
    
    choice = questionary.select(
        "请选择你的 LLM 提供商：",
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in BASE_URLS
        ],
        instruction="\n- 使用方向键导航\n- 按回车选择",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print("\n[red]未选择 OpenAI 后端，正在退出...[/red]")
        exit(1)

    display_name, url = choice
    print(f"你选择了：{display_name}\tURL：{url}")

    return display_name, url


def ask_openai_reasoning_effort() -> str:
    """选择 OpenAI 推理力度。"""
    choices = [
        questionary.Choice("Medium (Default)", "medium"),
        questionary.Choice("High (More thorough)", "high"),
        questionary.Choice("Low (Faster)", "low"),
    ]
    return questionary.select(
        "请选择推理力度：",
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """选择 Anthropic 的 effort 级别。

    用于控制 Claude 4.5+ 和 4.6 模型的 token 使用量和回答详尽程度。
    """
    return questionary.select(
        "请选择 effort 级别：",
        choices=[
            questionary.Choice("High (recommended)", "high"),
            questionary.Choice("Medium (balanced)", "medium"),
            questionary.Choice("Low (faster, cheaper)", "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """选择 Gemini 思考配置。

    返回 thinking_level: "high" 或 "minimal"。
    客户端会根据模型系列映射到对应的 API 参数。
    """
    return questionary.select(
        "请选择思考模式：",
        choices=[
            questionary.Choice("Enable Thinking (recommended)", "high"),
            questionary.Choice("Minimal/Disable Thinking", "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()


def ask_output_language() -> str:
    """选择报告输出语言。"""
    choice = questionary.select(
        "请选择输出语言：",
        choices=[
            questionary.Choice("English（默认）", "English"),
            questionary.Choice("Chinese (中文)", "Chinese"),
            questionary.Choice("Japanese (日本語)", "Japanese"),
            questionary.Choice("Korean (한국어)", "Korean"),
            questionary.Choice("Hindi (हिन्दी)", "Hindi"),
            questionary.Choice("Spanish (Español)", "Spanish"),
            questionary.Choice("Portuguese (Português)", "Portuguese"),
            questionary.Choice("French (Français)", "French"),
            questionary.Choice("German (Deutsch)", "German"),
            questionary.Choice("Arabic (العربية)", "Arabic"),
            questionary.Choice("Russian (Русский)", "Russian"),
            questionary.Choice("自定义语言", "custom"),
        ],
        style=questionary.Style([
            ("selected", "fg:yellow noinherit"),
            ("highlighted", "fg:yellow noinherit"),
            ("pointer", "fg:yellow noinherit"),
        ]),
    ).ask()

    if choice == "custom":
        return questionary.text(
            "请输入语言名称（例如：Turkish、Vietnamese、Thai、Indonesian）：",
            validate=lambda x: len(x.strip()) > 0 or "请输入语言名称。",
        ).ask().strip()

    return choice
