import json
import os
import datetime
import logging
from typing import List, Dict
import asyncio

from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import event_message_type, EventMessageType
from astrbot.api.star import Context, Star, register
from astrbot.api.event.filter import command, command_group, llm_tool, on_decorating_result
from sentence_transformers import SentenceTransformer, util
import httpx

logger = logging.getLogger("astrbot")


@register("ai_memory_longterm", "kjqwdw", "一个长久记忆插件", "1.0.0")
class Main(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.PLUGIN_NAME = "ai_memory_longterm"

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.summary_file = os.path.join(plugin_dir, "summary_data.json")
        self.memory_file = os.path.join(plugin_dir, "memory_data.json")

        if not os.path.exists(self.summary_file):
            with open(self.summary_file, "w", encoding='utf-8') as f:
                f.write("{}")

        if not os.path.exists(self.memory_file):
            with open(self.memory_file, "w", encoding='utf-8') as f:
                f.write("{}")

        with open(self.summary_file, "r", encoding='utf-8') as f:
            self.summaries = json.load(f)

        try:
            with open(self.memory_file, "r", encoding='utf-8') as f:
                self.memory_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.memory_data = {}
            with open(self.memory_file, "w", encoding='utf-8') as f:
                json.dump(self.memory_data, f, ensure_ascii=False)

        self.max_summaries = config.get("max_summaries", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.summary_model = config.get("summary_model", "paraphrase-multilingual-MiniLM-L12-v2")
        self.embedding_model = SentenceTransformer(self.summary_model, local_files_only=True)
        self.siliconflow_api_key = config.get("siliconflow_api_key", "")
        self.siliconflow_model = config.get("siliconflow_model", "Qwen/Qwen2.5-7B-Instruct")
        self.auto_summary_interval = config.get("auto_summary_interval", 10)  # 读取新的配置项

        self.locks = {}  # 用于存储每个会话的锁

    @command_group("memory")
    def memory(self):
        """记忆管理指令组"""
        pass

    @memory.command("list")
    async def list_summaries(self, event: AstrMessageEvent):
        """列出所有摘要"""
        session_id = self._get_unified_session_id(event)
        if session_id not in self.summaries or not self.summaries[session_id]:
            return event.plain_result("当前会话没有摘要。")

        summaries_text = "已保存的摘要:\n"
        for i, summary in enumerate(self.summaries[session_id]):
            summaries_text += f"{i + 1}. {summary['content']} (时间:{summary['timestamp']})\n"
        return event.plain_result(summaries_text)

    @memory.command("summary")
    async def create_summary(self, event: AstrMessageEvent):
        """总结并保存自上次总结以来的新对话"""
        session_id = self._get_unified_session_id(event)
        await self._create_summary_internal(session_id, event)

    @memory.command("clear")
    async def clear_summaries(self, event: AstrMessageEvent):
        """清空当前会话的所有摘要"""
        session_id = self._get_unified_session_id(event)
        if session_id in self.summaries:
            del self.summaries[session_id]
            await self._save_summaries()
            return event.plain_result("已清空所有摘要。")
        return event.plain_result("当前会话没有摘要。")

    @memory.command("remove")
    async def remove_summary(self, event: AstrMessageEvent, index: int):
        """删除指定序号的摘要"""
        session_id = self._get_unified_session_id(event)
        if session_id not in self.summaries:
            return event.plain_result("当前会话没有摘要。")

        summaries = self.summaries[session_id]
        index = index - 1
        if index < 0 or index >= len(summaries):
            return event.plain_result("无效的摘要序号。")

        removed = summaries.pop(index)
        await self._save_summaries()
        return event.plain_result(f"已删除摘要: {removed['content']}")

    @command("mem_help")
    async def memory_help(self, event: AstrMessageEvent):
        """显示记忆插件帮助信息"""
        help_text = f"""记忆插件使用帮助：

1. 记忆管理指令：
    /memory list - 列出所有摘要
    /memory summary - 总结并保存之前的对话
    /memory clear - 清空当前会话的所有摘要
    /memory remove <序号> - 删除指定序号的摘要
    /mem_help - 显示此帮助信息

2. 记忆特性：
    - 每个会话最多保存{self.max_summaries}条摘要
    - AI 在对话时会参考历史摘要, 实现长久记忆
    - 每{self.auto_summary_interval}轮对话自动进行一次总结
        """

        return event.plain_result(help_text)

    async def _get_lock(self, session_id: str) -> asyncio.Lock:
        """获取或创建会话锁"""
        if session_id not in self.locks:
            self.locks[session_id] = asyncio.Lock()
        return self.locks[session_id]

    async def _append_to_memory_data(self, session_id: str, role: str, content: str, event: AstrMessageEvent):
        """将消息追加到 memory_data.json，并更新计数器"""
        if session_id not in self.memory_data:
            self.memory_data[session_id] = {"messages": [], "count": 0, "last_summary_index": -1}

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.memory_data[session_id]["messages"].append(message)
        self.memory_data[session_id]["count"] += 1

        await self._save_memory_data()

        # 检查是否需要自动总结
        if self.memory_data[session_id]["count"] % self.auto_summary_interval == 0:
            await self._create_summary_internal(session_id, None)  # 自动总结，传入 None


    async def _save_memory_data(self):
        """保存 memory_data.json"""
        with open(self.memory_file, "w", encoding='utf-8') as f:
            json.dump(self.memory_data, f, ensure_ascii=False, indent=4)

    async def _save_summaries(self):
        """保存摘要到文件"""
        with open(self.summary_file, "w", encoding='utf-8') as f:
            json.dump(self.summaries, f, ensure_ascii=False, indent=4)

    def _get_unified_session_id(self, event: AstrMessageEvent) -> str:
        return event.unified_msg_origin

    async def _create_summary_internal(self, session_id: str, event: AstrMessageEvent):
        """内部方法：总结并保存自上次总结以来的新对话, 供手动和自动调用"""

        if session_id not in self.memory_data:
            self.memory_data[session_id] = {"messages": [], "count": 0, "last_summary_index": -1}

        last_summary_index = self.memory_data[session_id].get("last_summary_index", -1)
        new_messages = self.memory_data[session_id]["messages"][last_summary_index + 1:]

        if not new_messages:
            if event: # 手动调用时发送
                await event.send("没有新的对话需要总结。")
            return

        conversation_history = ""
        for entry in new_messages:
            conversation_history += f"{entry['role']}: {entry['content']}\n"

        summary_text = await self._summarize_text_with_siliconflow(conversation_history)

        if not summary_text:
            if event:  # 只有手动调用时才发送
                await event.send("总结失败：没有生成摘要。")
            return

        await self._save_summary(session_id, summary_text)
        self.memory_data[session_id]["last_summary_index"] = len(self.memory_data[session_id]["messages"]) - 1
        await self._save_memory_data()
        if event: # 手动调用时发送
            await event.send(f"已生成并保存对话摘要：\n{summary_text}")
        logger.info(f"会话 {session_id} 已总结：\n{summary_text}")

    async def _summarize_text_with_siliconflow(self, text: str) -> str:
        """使用硅基流动 API 总结文本"""

        if not text:
            return ""

        url = "https://api.siliconflow.cn/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.siliconflow_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.siliconflow_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"请对以下对话进行总结摘要：\n{text}\n摘要应简洁、清晰，并包含关键信息。"
                }
            ],
            "stream": False,
            "max_tokens": 512,
            "stop": ["null"],
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"},
            "tools": []
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                result = response.json()
                logger.debug(f"硅基流动 API 响应: {result}")

                if "choices" in result and result["choices"] and result["choices"][0].get("message", {}).get(
                        "content"):
                    summary = result["choices"][0]["message"]["content"]
                    return summary
                else:
                    error_message = result.get("message", "未知错误")
                    logger.error(f"硅基流动 API 错误: {error_message} 或 返回结果格式错误:{result}")
                    return f"总结失败: {error_message}"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 状态码错误: {e.response.status_code} - {e}")
            return f"总结失败：HTTP 状态码错误 - {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"HTTP 请求错误: {e}")
            return f"总结失败：HTTP 请求错误 - {e}"
        except Exception as e:
            logger.exception(f"总结时发生其他错误: {e}")
            return f"总结失败：{e}"


    async def _save_summary(self, session_id: str, summary_text: str):
        """保存摘要"""
        if session_id not in self.summaries:
            self.summaries[session_id] = []
        try:
            if len(self.summaries[session_id]) >= self.max_summaries:
                self.summaries[session_id].pop(0)
        except Exception as e:
            logger.exception(f"删除旧摘要时出错: {e}")

        summary = {
            "content": summary_text,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.summaries[session_id].append(summary)
        await self._save_summaries()

    async def _get_relevant_summaries(self, session_id: str, query: str) -> List[str]:
        """根据相似度获取相关的摘要"""
        if session_id not in self.summaries or not self.summaries[session_id]:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        relevant_summaries = []

        for summary in self.summaries[session_id]:
            summary_embedding = self.embedding_model.encode(summary['content'], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, summary_embedding)[0][0]

            if similarity >= self.similarity_threshold:
                relevant_summaries.append((summary['content'], similarity))

        relevant_summaries.sort(key=lambda x: x[1], reverse=True)
        return [summary for summary, _ in relevant_summaries[:3]]

    @llm_tool(name="get_memories")
    async def get_memories(self, event: AstrMessageEvent, current_input: str = "") -> str:
        """LLM工具：获取相关记忆"""
        session_id = self._get_unified_session_id(event)
        relevant_summaries = await self._get_relevant_summaries(session_id, current_input)
        if relevant_summaries:
            return "💭 相关摘要：\n" + "\n".join([f"- {s}" for s in relevant_summaries])

        return "我没有任何相关记忆。"

    @on_decorating_result()
    async def _record_all_messages(self, event: AstrMessageEvent):
        session_id = self._get_unified_session_id(event)
        curr_cid = await self.context.conversation_manager.get_curr_conversation_id(event.unified_msg_origin)
        if not curr_cid:
            return
        conversation = await self.context.conversation_manager.get_conversation(event.unified_msg_origin, curr_cid)
        if not conversation:
            return

        history = json.loads(conversation.history)
        if len(history) >= 2:
            last_user_message = history[-2]
            last_ai_message = history[-1]

            if last_user_message:
                await self._append_to_memory_data(session_id, last_user_message['role'], last_user_message['content'], event)
            if last_ai_message:
                await self._append_to_memory_data(session_id, last_ai_message['role'], last_ai_message['content'], event)