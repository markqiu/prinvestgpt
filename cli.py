import importlib

import typer
import os
import platform
from dotenv import load_dotenv
from prinvestGPT import VERSION

load_dotenv()  # take environment variables from .env.

app = typer.Typer()

if platform.system().lower() == "windows":
    OSTYPE = "windows"
elif platform.system().lower() == "linux":
    OSTYPE = "linux"
elif platform.system().lower() == "darwin":
    OSTYPE = "macos"


@app.command()
def version():
    """
    显示版本号
    """
    print(VERSION)


@app.command()
def build(name: str = "."):
    """
    编译打包
    """
    print(f"开始编译打包{name}，将在dist目录保存...")
    build_module = importlib.import_module("build")
    if not build_module:
        print("先安装build模块...")
        os.system("pip install build")
    os.system(f"python -m build -n {name}")


@app.command()
def test(name: str = "."):
    """
    运行pytest
    """
    print("开始运行pytest...")
    os.system(f"pytest {name}")


@app.command()
def export_corpus(export_format: str = "jsonl"):
    """
    更新环境
    """
    print(f"开始安装环境{name}...")
    if OSTYPE == "windows":
        os.system(f"pip install -U -e .[{name}]")
    else:
        os.system(f"pip install -U -e '.[{name}]'")


if __name__ == "__main__":
    app()
