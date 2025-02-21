# Memory Bank 插件

作者: Arain

版本: 1.1

描述: 为 AstrBot 提供永久记忆功能, 允许机器人记住之前的对话, 并在后续对话中参考这些记忆。

# 功能

记忆存储: 将对话历史总结为摘要，并存储起来。
记忆检索: 在对话中，根据当前输入检索相关的历史摘要。
自动总结: 每隔一定数量的对话轮次，自动总结新的对话内容。
手动总结: 使用指令手动总结对话。
记忆管理: 允许用户列出、删除和清除已保存的摘要。
LLM 工具集成: 作为 LLM 工具，可以利用 LLM 对记忆进行总结摘要，并在需要时使用记忆。

# 指令

该插件提供以下指令：

/memory list: 列出当前会话的所有摘要。
/memory summary: 总结并保存自上次总结以来的新对话。
/memory clear: 清空当前会话的所有摘要。
/memory remove <序号>: 删除指定序号的摘要。
/mem_help: 显示插件的帮助信息。

# 配置

您可以在插件配置中调整以下参数：

max_summaries: 每个会话最多保存的摘要数量，默认为 5。
similarity_threshold: 用于判断摘要与当前输入相关性的相似度阈值，默认为 0.7。
summary_mode: 用于生成摘要和计算文本相似度的模型，默认为 "paraphrase-multilingual-MiniLM-L12-v2"。
siliconflow_api_key: 硅基流动 API 的密钥，用于进行文本总结。
siliconflow_model: 硅基流动 API 使用的模型，默认为 "Qwen/Qwen2.5-7B-Instruct"。
auto_summary_interval: 自动总结的对话轮次间隔，默认为 10。

# 工作原理

1.消息记录: 插件会记录所有用户和机器人的消息。

2.自动/手动总结:
  每隔 auto_summary_interval 轮对话，插件会自动总结自上次总结以来的新对话。
  用户也可以使用/memory summary指令手动触发总结。
  总结使用硅基流动 API 完成。

3.摘要存储: 总结后的摘要会与时间戳一起保存。每个会话最多保存 max_summaries 条摘要，超出时会删除最早的摘要。

4. 记忆检索:
   当用户输入新的内容时，插件会计算输入与所有摘要的相似度 (使用 summary_model 配置的模型)。
   相似度高于 similarity_threshold 的摘要被认为是相关的。
   最多返回 3 条最相关的摘要。

5.  LLM 工具:插件提供 get_memories LLM 工具。当 LLM 需要历史信息时，可以调用此工具获取相关摘要。

# 依赖

  sentence_transformers
  httpx

请确保已安装这些依赖。

# 注意事项
本插件使用了硅基流动免费 API，请确保网络可以访问该服务，并且API密钥填写正确。当然如果您愿意也可以换成其它API，经测试7B模型的总结能力并不强...但由于总结摘要需要消耗大量tokens，这依然是可忍受的（富哥除外）。
安装流程：
直接在管理页面加载zip就可以完成安装。然后在XX盘\AstrBotLauncher-0.1.5.5\AstrBotLauncher-0.1.5.5\AstrBot下的requirements后添加httpx和sentence-transformers即可添加依赖。重新启动Astrbot将自动下载依赖，完成安装。

# 关于记忆
记忆位置存放在XX盘\AstrBotLauncher-0.1.5.5\AstrBotLauncher-0.1.5.5\AstrBot\data\plugins\memory_plugin下，聊天记录为memory_data.json文件，也就是插件配置中的“记忆”；总结摘要为summary_data.json文件，也就是插件配置中的“摘要”。
