from typing import Dict, Any


class ToolExecutor:
    """
    工具執行器，負責管理和執行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, description: str, func: callable):
        """
        向工具箱中註冊一個新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，將被覆蓋。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已註冊。")

    def get_tool(self, name: str) -> callable:
        """
        根據名稱取得一個工具的執行函式。
        """
        return self.tools.get(name, {}).get("func")

    def get_available_tools(self) -> str:
        """
        取得所有可用工具的格式化描述字串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
