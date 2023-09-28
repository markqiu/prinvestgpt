# prinvestGPT
原则投资知识库接入GPT

## 安装
基于hatch
```shell
hatch env create
```
或者用conda创建好基础环境，用hatch来管理更新
```shell
conda create -n prinvestGPT python=3.10
```

### 安装开发环境
自带的命令行工具pv的用法如下：
```shell
# 安装开发环境
hatch run pip install -e .[all]
```
## 使用
1. 如果要部署server，并且需要采用本地知识库，则需要先部署meilisearch。
```shell
# 查看帮助
pv -h
# 导出帮助中心
pv corpus export --fmt [txt|jsonl]
# 启动投资原则问答服务
pv server start
```

## 配置
配置文件在new_cof.yaml中，主要包括：
```yaml
API:
  ernie_client_id: 百度千帆的应用接入id
  ernie_client_secret: 百度千帆的应用接入secret
  openai_api_key: openai的api_key

MODELS:
  llm_model:  # 语言模型名称
    - Ernie  # 百度文心一言
    - OpenAI  # OpenAI的GPT3
  embedding_model:  # embedding模型名称
    - Ernie  # 百度文心一言
    - OpenAI  # OpenAI的GPT3

block:  # gradio服务配置
  concurrency_count: 32
  server_name: "0.0.0.0"
  server_port: 7860
  debug: true

meilisearch:  # meilisearch配置
  url: http://127.0.0.1:7700
  api_key: meilisearch的api_key

KNOWLEDGE_BASE:
  baklib:
    token : baklib帮助中心的token
    tenant_id : baklib帮助中心的站点id
```