from pprint import pprint
from pydantic import BaseModel, Field
from typing import Optional
from tqdm import tqdm
import jionlp as jio
import requests

BASE_URI = "https://www.baklib.com/api/v1"

token = "c0ab1713043e46aa2e38b2d1b772bb8ff2aa92b3ee6239f7e200e3c3f858aca2"
tenant_id = "a6b89e1d-1764-4fc0-8ec6-510e1250b050"


def get_channels(
        parent_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
) -> dict:
    """
    获取baklib栏目列表

    Parameters
    ----------
    parent_id: 父栏目id
    page: 需要显示的页码, 默认1
    per_page: 每页显示数量, 默认10, 最大不能超过50
    """
    response = requests.get(
        f"{BASE_URI}/channels",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "tenant_id": tenant_id,
            "parent_id": parent_id,
            "_page": page,
            "_per_page": per_page,
        },
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError:
        raise ValueError(f"获取balib文章列表失败: [{response_json['error']}]")


def get_articles(
        channel_id: Optional[str] = None,
        name: Optional[str] = None,
        identifier: Optional[str] = None,
        page: int = 1,
        per_page: int = 10,
) -> dict:
    """
    获取baklib文章列表

    Parameters
    ----------
    channel_id: 栏目id
    name: 通过标题搜索
    identifier: 通过identifier搜索
    page: 需要显示的页码, 默认1
    per_page: 每页显示数量, 默认10, 最大不能超过50

    Returns
    -------

    """
    response = requests.get(
        f"{BASE_URI}/articles",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "tenant_id": tenant_id,
            "channel_id": channel_id,
            "name": name,
            "identifier": identifier,
            "_page": page,
            "_per_page": per_page,
        },
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError:
        raise ValueError(f"获取balib文章列表失败: [{response_json['error']}]")


def get_articles_content(content_id: str) -> dict:
    # 根据文章id，查询文章具体内容
    response = requests.get(
        f"{BASE_URI}/articles/{content_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"tenant_id": tenant_id},
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError:
        raise ValueError(f"获取balib文章列表失败: [{response_json['error']}]")


def prompt_constuctor(question: str) -> str:
    return f"""
    作为一个投资原则方面的专家，请用严谨的语气进行说明，如果不知道请回答不知道。我的问题是：
    {question}
    """


class Knowledge(BaseModel):
    prompt: str
    response: str


if __name__ == "__main__":
    # 返回所有文章列表
    # pprint(get_articles()["meta"])
    # 返回指定栏目ID的文章列表
    # print(get_articles(channel_id="dda792f7-ca9f-4f23-bb55-f8735a42f8f4"))
    # 返回指定文章标题的文章列表
    # print(get_articles(name="在能力圈内投资"))

    # 根据文章id，查询文章内容
    # pprint(
    #     get_articles_content(content_id="bcbaad2c-9b72-455c-8779-d8f89f9c439a")
    # )
    meta_info = get_articles()['meta']
    current_page = meta_info['current_page']
    total_pages = meta_info['total_pages']
    total_count = meta_info['total_count']
    pbar = tqdm(desc="帮助中心导出进度", total=total_count, unit="篇")
    with open("baklib.jsonl", "w", encoding="utf8") as f:
        while current_page <= total_pages:
            for item in get_articles(page=current_page)['items']:
                current_article = get_articles_content(item['id'])
                blocks = current_article['content']["blocks"]
                content_str = "\n".join([block['data']['text'] for block in blocks if "text" in block['data']])
                k = Knowledge(prompt=prompt_constuctor(current_article['name']), response=jio.clean_text(content_str))
                f.write(k.model_dump_json())
                f.write("\n")
                pbar.update(1)
            current_page += 1
    pbar.close()
