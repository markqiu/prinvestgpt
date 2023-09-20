import requests
import yaml

BASE_URI = "https://www.baklib.com/api/v1"


with open("./new_cof.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
token = config["KNOWLEDGE_BASE"]["baklib"]["token"]
tenant_id = config["KNOWLEDGE_BASE"]["baklib"]["tenant_id"]


def get_channels(
    parent_id: str | None = None,
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
        timeout=30,
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError as err:
        msg = f"获取balib文章列表失败: [{response_json['error']}]"
        raise ValueError(msg) from err


def get_articles(
    channel_id: str | None = None,
    name: str | None = None,
    identifier: str | None = None,
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
        timeout=30,
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError as err:
        msg = f"获取balib文章列表失败: [{response_json['error']}]"
        raise ValueError(msg) from err


def get_articles_content(content_id: str) -> dict:
    # 根据文章id,查询文章具体内容
    response = requests.get(
        f"{BASE_URI}/articles/{content_id}",
        headers={"Authorization": f"Bearer {token}"},
        params={"tenant_id": tenant_id},
        timeout=30,
    )
    response_json = response.json()
    try:
        return response_json["message"]
    except KeyError as err:
        msg = f"获取balib文章列表失败: [{response_json['error']}]"
        raise ValueError(msg) from err


def prompt_constructor(question: str) -> str:
    return f"作为一个投资原则方面的专家, 请用严谨的语气进行说明，如果不知道请回答不知道。我的问题是：{question}"
