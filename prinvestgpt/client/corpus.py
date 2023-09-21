import json

import jionlp as jio
import typer
from tqdm import tqdm

from prinvestgpt.knowledge_base.sources.baklib.baklib_api import (
    get_articles,
    get_articles_content,
)

app = typer.Typer()


@app.command()
def export(fmt: str = "jsonl", include: str | None = None, exclude: str | None = None):
    """
    导出语料库
    TODO 名词解释
    TODO 成长教材
    TODO 原则库、助手库
    """
    meta_info = get_articles()["meta"]
    current_page = meta_info["current_page"]
    total_pages = meta_info["total_pages"]
    total_count = meta_info["total_count"]
    pbar = tqdm(desc="帮助中心导出进度", total=total_count, unit="篇")
    if fmt == "jsonl":
        with open("baklib.jsonl", "w", encoding="utf8") as outfile:
            while current_page <= total_pages:
                for item in get_articles(page=current_page)["items"]:
                    current_article = get_articles_content(item["id"])
                    blocks = current_article["content"]["blocks"]
                    k = [
                        {
                            "prompt": current_article["name"],
                            "response": [
                                [
                                    "\n".join(
                                        [
                                            jio.clean_text(block["data"]["text"])
                                            for block in blocks
                                            if "text" in block["data"]
                                        ]
                                    )
                                ]
                            ],
                        }
                    ]
                    outfile.write(json.dumps(k, ensure_ascii=False) + "\n")
                    pbar.update(1)
                current_page += 1
    elif fmt == "txt":
        with open("baklib.txt", "w", encoding="utf8") as outfile:
            while current_page <= total_pages:
                for item in get_articles(page=current_page)["items"]:
                    current_article = get_articles_content(item["id"])
                    blocks = current_article["content"]["blocks"]
                    block_splitter = ","
                    item_splitter = "\n\n\n"
                    outfile.write(current_article["name"] + block_splitter)
                    outfile.write(
                        block_splitter.join(
                            [jio.clean_text(block["data"]["text"]) for block in blocks if "text" in block["data"]]
                        )
                        + item_splitter
                    )
                    pbar.update(1)
                current_page += 1
    else:
        msg = f"不支持的导出格式: {fmt}"
        raise ValueError(msg)
    pbar.close()


if __name__ == "__main__":
    app()
