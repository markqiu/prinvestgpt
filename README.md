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

## 使用
```shell
# 查看帮助
pv -h
# 导出帮助中心
pv corpus export --fmt txt|jsonl
# 启动投资原则问答服务
pv server start
```

