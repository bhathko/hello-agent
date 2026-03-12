"""
Main entry point for the Hello-Agent application.
"""
import argparse

# Import example runners
from examples.transformer_structure import main as run_transformer_structure
from examples.action_thought_observe import main as run_action_thought_observe
from examples.classic_agent_build import main as run_classic_agent_build
from examples.basic_agent_tool import main as run_basic_agent_tool

EXAMPLES = {
    1: ("Transformer 結構介紹", run_transformer_structure),
    2: ("Action-Thought-Observe ReAct 智慧體", run_action_thought_observe),
    3: ("經典 LLM Agent 建構", run_classic_agent_build),
    4: ("基礎智慧體工具使用", run_basic_agent_tool),
}


def main():
    """Main application entry point with chapter selection."""
    # 組裝可用範例的說明文字
    examples_help = "\n".join(
        f"  {num}. {desc}" for num, (desc, _) in EXAMPLES.items()
    )

    parser = argparse.ArgumentParser(
        description="Hello-Agent：逐章探索 AI 智慧體",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"可用範例：\n{examples_help}",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        choices=list(EXAMPLES.keys()),
        default=4,
        help="要執行的範例編號（預設：4）",
    )

    args = parser.parse_args()

    desc, runner = EXAMPLES.get(args.chapter, (None, None))
    if runner:
        print(f"▶ 執行範例 {args.chapter}：{desc}\n")
        runner()
    else:
        print(f"找不到範例 {args.chapter}。")


if __name__ == "__main__":
    main()
