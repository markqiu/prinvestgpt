import json
import logging

import requests

API_KEY = "hvY28gKyNPqTkrYYXAmsVrCz"
SECRET_KEY = "Vw0uKv3b61tXjIGQWOTHMQC3XYZcHxHs"


def main():
    url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/plugin/principle_invest/?access_token="
        + get_access_token()
    )

    question = "如何投资"
    payload = json.dumps(
        {"query": f"你现在的角色是一名投资经理，我是你的顾客，我想知道{question}, 请你结合长期发展、通货膨胀，风险预估、投资行业推荐介绍，来帮我定制最佳的投资计划。"}
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    logging.info(response.text)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": API_KEY,
        "client_secret": SECRET_KEY,
    }
    return str(requests.post(url, params=params, timeout=10).json().get("access_token"))


if __name__ == "__main__":
    main()
