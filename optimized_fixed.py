#!/usr/bin/env python3
"""
修复版优化语音增强器 - 集成OpenAI高级记忆系统
"""
import asyncio
import signal
import sys
import time
import subprocess
import threading
import uuid
import re
import os
import aiohttp
import json
from queue import Queue
from loguru import logger
from config import config
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

class FixedTTSEngine:
    """修复的TTS引擎"""
    
    def __init__(self):
        self.audio_buffer = asyncio.Queue(maxsize=20)  # 减小缓冲区
        self.tts_semaphore = asyncio.Semaphore(3)  # 增加到3个并发
        self.running = True
        self.session_players = {}  # 每个会话的播放器
        logger.info("🔊 修复TTS引擎初始化")
    
    async def start_engine(self):
        """启动TTS引擎"""
        asyncio.create_task(self._audio_player())
        logger.info("✅ 修复TTS引擎已启动")
    
    async def speak_streaming(self, text, session_id=None):
        """修复的流式语音合成"""
        try:
            if not self.running or not text.strip():
                return
                
            if not session_id:
                session_id = str(uuid.uuid4())[:8]
            
            # 更简单的分句策略
            sentences = self._simple_split_sentences(text)
            if not sentences:
                return
                
            logger.info(f"🎵 开始TTS[{session_id}]: {len(sentences)}句")
            
            # 顺序处理，避免并发问题
            for i, sentence in enumerate(sentences):
                if not self.running:
                    break
                    
                try:
                    await self._process_single_sentence(sentence, session_id, i)
                except Exception as e:
                    logger.error(f"处理句子失败[{session_id}-{i}]: {e}")
                    continue
            
            logger.info(f"✅ TTS完成[{session_id}]")
            
        except Exception as e:
            logger.error(f"流式TTS失败: {e}")
            logger.info(f"📝 文字回答: {text}")
    
    def _simple_split_sentences(self, text):
        """简化的分句策略"""
        if not text.strip():
            return []
        
        # 基本的句子分割
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in ['。', '！', '？', '!', '?'] and len(current.strip()) > 3:
                sentences.append(current.strip())
                current = ""
            elif char in ['，', ','] and len(current.strip()) > 20:
                sentences.append(current.strip())
                current = ""
        
        # 添加剩余部分
        if current.strip() and len(current.strip()) > 3:
            sentences.append(current.strip())
        
        return sentences
    
    async def _process_single_sentence(self, sentence, session_id, index):
        """处理单个句子"""
        async with self.tts_semaphore:
            try:
                task_id = f"{session_id}-{index}"
                logger.debug(f"🔊 TTS[{task_id}]: {sentence}")
                
                # 生成音频文件
                audio_file = await self._generate_audio_file(sentence)
                if not audio_file or not self.running:
                    return
                
                # 立即播放
                await self._play_audio_immediately(audio_file, task_id, sentence)
                
            except Exception as e:
                logger.error(f"处理句子失败: {e}")
    
    async def _generate_audio_file(self, text):
        """生成音频文件"""
        try:
            import edge_tts
            import tempfile
            
            # 使用更稳定的语音
            communicate = edge_tts.Communicate(
                text, 
                "zh-CN-XiaoxiaoNeural",  # 使用更稳定的声音
                rate="+20%"  # 减少语速调整
            )
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            # 生成音频，添加超时
            await asyncio.wait_for(
                communicate.save(temp_file.name),
                timeout=10.0  # 10秒超时
            )
            
            return temp_file.name
            
        except asyncio.TimeoutError:
            logger.warning(f"TTS生成超时: {text}")
            return None
        except Exception as e:
            logger.error(f"生成音频失败: {e}")
            return None
    
    async def _play_audio_immediately(self, audio_file, task_id, text):
        """立即播放音频"""
        try:
            if not self.running:
                return
                
            logger.info(f"🎵 播放[{task_id}]: {text}")
            
            if sys.platform == "darwin":  # macOS
                # 使用异步子进程
                process = await asyncio.create_subprocess_exec(
                    "afplay", audio_file,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                
                try:
                    # 等待播放完成，但设置超时
                    await asyncio.wait_for(process.wait(), timeout=15.0)
                    logger.debug(f"✅ 播放完成[{task_id}]")
                except asyncio.TimeoutError:
                    process.terminate()
                    logger.warning(f"⏰ 播放超时[{task_id}]")
            else:
                logger.info(f"💾 音频文件: {audio_file}")
            
            # 延迟删除文件
            asyncio.create_task(self._cleanup_audio_file(audio_file))
            
        except Exception as e:
            logger.error(f"播放音频失败[{task_id}]: {e}")
    
    async def _audio_player(self):
        """简化的音频播放器（备用）"""
        while self.running:
            await asyncio.sleep(1)  # 基本的保活循环
    
    async def _cleanup_audio_file(self, file_path):
        """清理音频文件"""
        try:
            await asyncio.sleep(2)  # 等待播放完成
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.debug(f"清理文件失败: {e}")
    
    async def stop(self):
        """停止TTS引擎"""
        logger.info("🛑 停止TTS引擎...")
        self.running = False

class FixedVoiceEnhancer:
    """修复版语音增强器 - 集成OpenAI记忆系统"""
    
    def __init__(self):
        self.running = False
        self.recognizer = None
        self.openai_client = None
        
        # 修复的TTS引擎
        self.tts_engine = FixedTTSEngine()
        
        # 简化的队列系统
        self.partial_text_queue = Queue(maxsize=20)
        self.final_text_queue = Queue(maxsize=10)
        self.response_queue = asyncio.Queue(maxsize=5)
        
        # 并发控制
        self.response_semaphore = asyncio.Semaphore(2)
        self.active_responses = {}
        
        # 修复的思考系统
        self.thinking_tasks = {}
        self.partial_thoughts = {}
        self.last_thinking_time = {}
        self.thinking_cooldown = 1.0  # 增加冷却时间
        
        # 对话状态
        self.speech_segments = []
        self.last_speech_time = 0
        self.silence_threshold = 2.0  # 增加静音阈值
        
        # === 新增：OpenAI记忆系统 ===
        self.openai_memory_manager = None
        self.memory_enabled = True
        
        logger.info("🚀 修复版语音增强器初始化 (支持OpenAI记忆)")
    
    async def start(self):
        """启动系统"""
        logger.info("🚀 启动修复版语音增强器...")
        
        try:
            await self._init_speech_recognition()
            await self._init_openai()
            await self._init_openai_memory()
            await self.tts_engine.start_engine()
            
            self.running = True
            logger.info("✅ 系统启动成功！")
            logger.info("")
            logger.info("🎤 = = = OpenAI记忆版功能 = = =")
            logger.info("🧠 OpenAI智能记忆系统")
            logger.info("🔍 向量相似度搜索")
            logger.info("⚡ GPT智能信息提取")
            logger.info("🎯 智能冲突解决")
            logger.info("💬 个性化回答生成")
            logger.info("⌨️  按 Ctrl+C 退出")
            logger.info("= = = = = = = = = = = = =")
            logger.info("")
            
            # 启动处理任务
            self.processing_tasks = []
            self.processing_tasks.append(asyncio.create_task(self._partial_text_processor(), name="partial"))
            self.processing_tasks.append(asyncio.create_task(self._final_text_processor(), name="final"))
            self.processing_tasks.append(asyncio.create_task(self._response_handler(), name="response"))
            self.processing_tasks.append(asyncio.create_task(self._cleanup_manager(), name="cleanup"))

            logger.info(f"✅ 启动了 {len(self.processing_tasks)} 个处理任务")
            
            # 信号处理
            def signal_handler(signum, frame):
                logger.info("🛑 收到退出信号")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            try:
                while self.running:
                    await asyncio.sleep(1)
                    
                    # 检查任务状态
                    for task in self.processing_tasks:
                        if task.done() and not task.cancelled():
                            exception = task.exception()
                            if exception:
                                logger.error(f"任务 {task.get_name()} 异常: {exception}")
                                # 重启失败的任务
                                await self._restart_failed_task(task)
                
            except Exception as e:
                logger.error(f"主循环错误: {e}")
            finally:
                await self._graceful_shutdown()
                
        except Exception as e:
            logger.error(f"启动失败: {e}")
            await self.stop()
    
    async def _restart_failed_task(self, failed_task):
        """重启失败的任务"""
        try:
            task_name = failed_task.get_name()
            logger.info(f"🔄 重启任务: {task_name}")
            
            # 移除失败的任务
            self.processing_tasks.remove(failed_task)
            
            # 重新创建任务
            if task_name == "partial":
                new_task = asyncio.create_task(self._partial_text_processor(), name="partial")
            elif task_name == "final":
                new_task = asyncio.create_task(self._final_text_processor(), name="final")
            elif task_name == "response":
                new_task = asyncio.create_task(self._response_handler(), name="response")
            elif task_name == "cleanup":
                new_task = asyncio.create_task(self._cleanup_manager(), name="cleanup")
            else:
                return
            
            self.processing_tasks.append(new_task)
            logger.info(f"✅ 任务重启成功: {task_name}")
            
        except Exception as e:
            logger.error(f"重启任务失败: {e}")
    
    async def _graceful_shutdown(self):
        """优雅关闭"""
        logger.info("🛑 开始优雅关闭...")
        self.running = False
        
        # 取消所有任务
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.processing_tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning("部分任务未能及时完成")
        
        await self.stop()
        logger.info("✅ 优雅关闭完成")
    
    async def _init_speech_recognition(self):
        """初始化语音识别"""
        logger.info("🎙️ 初始化语音识别...")
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=config.azure.speech_key,
                region=config.azure.speech_region
            )
            speech_config.speech_recognition_language = "zh-CN"
            
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self.recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            def on_recognizing(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                    partial_text = evt.result.text.strip()
                    if partial_text and len(partial_text) >= 4:
                        try:
                            if not self.partial_text_queue.full():
                                self.partial_text_queue.put_nowait({
                                    'text': partial_text,
                                    'timestamp': time.time(),
                                    'id': str(uuid.uuid4())[:8]
                                })
                        except:
                            pass
            
            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    final_text = evt.result.text.strip()
                    if final_text and len(final_text) > 1:
                        try:
                            if not self.final_text_queue.full():
                                self.final_text_queue.put_nowait({
                                    'text': final_text,
                                    'timestamp': time.time(),
                                    'id': str(uuid.uuid4())[:8]
                                })
                        except:
                            pass
            
            self.recognizer.recognizing.connect(on_recognizing)
            self.recognizer.recognized.connect(on_recognized)
            
            self.recognizer.start_continuous_recognition()
            logger.info("✅ 语音识别已启动")
            
        except Exception as e:
            logger.error(f"语音识别初始化失败: {e}")
            raise
    
    async def _init_openai(self):
        """初始化OpenAI"""
        logger.info("🤖 初始化OpenAI...")
        
        try:
            import openai
            self.openai_client = openai.AsyncOpenAI(api_key=config.openai.api_key)
            
            # 测试连接
            test_response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "测试"}],
                max_tokens=5
            )
            
            if test_response.choices:
                logger.info("✅ OpenAI连接成功")
            else:
                raise Exception("OpenAI测试失败")
                
        except Exception as e:
            logger.error(f"OpenAI初始化失败: {e}")
            raise
    
    async def _init_openai_memory(self):
        """初始化OpenAI记忆系统"""
        try:
            logger.info("🧠 初始化OpenAI记忆系统...")
            
            from openai_memory_manager import OpenAIMemoryManager
            
            self.openai_memory_manager = OpenAIMemoryManager(
                openai_client=self.openai_client,
                data_dir="data/openai_memory"
            )
            
            # 显示记忆统计
            stats = await self.openai_memory_manager.get_memory_stats()
            logger.info(f"📊 记忆统计: {stats['total_memories']} 条记忆")
            
            if stats['by_type']:
                type_info = ", ".join([f"{k}: {v}" for k, v in stats['by_type'].items()])
                logger.info(f"📋 记忆类型分布: {type_info}")
            
            self.memory_enabled = True
            logger.info("✅ OpenAI记忆系统初始化成功")
            
        except Exception as e:
            logger.error(f"OpenAI记忆系统初始化失败: {e}")
            self.memory_enabled = False
            self.openai_memory_manager = None
    
    async def _partial_text_processor(self):
        """部分文本处理器"""
        while self.running:
            try:
                if not self.partial_text_queue.empty():
                    data = self.partial_text_queue.get_nowait()
                    await self._handle_partial_text(data)
                else:
                    await asyncio.sleep(0.05)
            except Exception as e:
                if self.running:
                    logger.error(f"处理部分文本错误: {e}")
                await asyncio.sleep(0.2)
    
    async def _handle_partial_text(self, data):
        """处理部分文本"""
        try:
            partial_text = data['text']
            text_id = data['id']
            timestamp = data['timestamp']
            
            # 冷却时间检查
            if text_id in self.last_thinking_time:
                if timestamp - self.last_thinking_time[text_id] < self.thinking_cooldown:
                    return
            
            logger.debug(f"🎧 实时[{text_id}]: {partial_text}")
            
            if self._should_start_thinking(partial_text):
                logger.info(f"🧠 构思[{text_id}]: {partial_text}")
                self.last_thinking_time[text_id] = timestamp
                
                # 清理旧的思考任务
                await self._cleanup_old_thinking_tasks(text_id)
                
                # 启动新的思考
                if text_id not in self.thinking_tasks:
                    self.thinking_tasks[text_id] = asyncio.create_task(
                        self._smart_thinking(partial_text, text_id)
                    )
                
        except Exception as e:
            logger.error(f"处理部分文本失败: {e}")
    
    async def _cleanup_old_thinking_tasks(self, current_id):
        """清理旧的思考任务"""
        try:
            # 保留最近的2个任务
            task_ids = list(self.thinking_tasks.keys())
            if len(task_ids) > 2:
                for old_id in task_ids[:-2]:
                    if old_id != current_id and old_id in self.thinking_tasks:
                        self.thinking_tasks[old_id].cancel()
                        del self.thinking_tasks[old_id]
        except Exception as e:
            logger.debug(f"清理思考任务失败: {e}")
    
    def _should_start_thinking(self, text):
        """判断是否开始思考"""
        if len(text) < 6:
            return False
        
        triggers = [
            '什么', '怎么', '为什么', '哪里', '谁', '如何',
            '能不能', '可以', '会不会', '是不是'
        ]
        
        questions = ['？', '?', '吗', '呢']
        
        has_trigger = any(t in text for t in triggers)
        has_question = any(q in text for q in questions)
        
        return has_trigger or has_question
    
    async def _smart_thinking(self, partial_text, text_id):
        """智能思考"""
        try:
            prompt = f"用户可能想问：'{partial_text}' 简单预判（5字内）："
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.7
            )
            
            if response.choices:
                thought = response.choices[0].message.content.strip()
                self.partial_thoughts[text_id] = {
                    'thought': thought,
                    'partial_text': partial_text,
                    'timestamp': time.time()
                }
                logger.info(f"⚡ 预判[{text_id}]: {thought}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"思考失败[{text_id}]: {e}")
        finally:
            # 清理任务引用
            if text_id in self.thinking_tasks:
                del self.thinking_tasks[text_id]
    
    async def _final_text_processor(self):
        """最终文本处理器"""
        while self.running:
            try:
                if not self.final_text_queue.empty():
                    data = self.final_text_queue.get_nowait()
                    await self._handle_final_text(data)
                else:
                    await asyncio.sleep(0.05)
            except Exception as e:
                if self.running:
                    logger.error(f"处理完整文本错误: {e}")
                await asyncio.sleep(0.2)
    
    def _check_force_record_keywords(self, text: str) -> bool:
        """检查是否包含强制记录关键词"""
        force_keywords = [
            "记录一下", "记住", "记下来", "保存一下", 
            "添加到记忆", "记录", "存储", "记下",
            "别忘了", "要记住", "帮我记一下"
        ]
        
        return any(keyword in text for keyword in force_keywords)

    def _extract_content_after_keywords(self, text: str) -> str:
        """提取强制记录关键词后的内容"""
        force_keywords = ["记录一下", "记住", "记下来", "保存一下", "记录", "记下"]
        
        for keyword in force_keywords:
            if keyword in text:
                # 找到关键词位置，提取后面的内容
                index = text.find(keyword) + len(keyword)
                content = text[index:].strip()
                if content:
                    return content
        
        return text  # 如果没找到关键词，返回原文

    async def _handle_final_text(self, data):
        """处理完整文本"""
        try:
            final_text = data['text']
            text_id = data['id']
            timestamp = data['timestamp']
            
            # 检查对话段落
            if timestamp - self.last_speech_time > self.silence_threshold:
                logger.info("🔄 新对话段落")
                self.speech_segments = []
            
            self.speech_segments.append({
                'text': final_text,
                'timestamp': timestamp,
                'id': text_id
            })
            
            self.last_speech_time = timestamp
            logger.info(f"🗣️  完整[{text_id}]: {final_text}")
            
            # === OpenAI记忆处理 ===
            if self.memory_enabled and self.openai_memory_manager:
                try:
                    # 检查是否包含强制记录关键词
                    force_record = self._check_force_record_keywords(final_text)
                    
                    # 异步处理记忆存储，不阻塞主流程
                    asyncio.create_task(self._process_memory(final_text, force_record=force_record))
                except Exception as e:
                    logger.debug(f"记忆处理失败: {e}")
            
            if self._should_respond(final_text):
                logger.info(f"🤖 需要回应[{text_id}]")
                
                context_text = self._build_context()
                prior_thought = self.partial_thoughts.get(text_id, {})
                
                try:
                    await self.response_queue.put({
                        'text': context_text,
                        'final_text': final_text,
                        'timestamp': timestamp,
                        'id': text_id,
                        'prior_thought': prior_thought
                    })
                except asyncio.QueueFull:
                    logger.warning(f"响应队列满[{text_id}]")
            
            # 清理思考任务
            if text_id in self.thinking_tasks:
                self.thinking_tasks[text_id].cancel()
                del self.thinking_tasks[text_id]
            
        except Exception as e:
            logger.error(f"处理完整文本失败: {e}")
    
    async def _process_memory(self, text: str, force_record: bool = False):
        """异步处理记忆 - 添加AI回答过滤和强制记录功能"""
        try:
            context = self._build_context()  # 提前定义context
            
            if force_record:
                # 强制记录模式：提取关键词后的内容
                cleaned_text = self._extract_content_after_keywords(text)
                success = await self.openai_memory_manager.force_store(cleaned_text, context)
                if success:
                    logger.info(f"🔒 强制记录: {cleaned_text}")
                return
            
            # 检查是否是用户在复述AI的回答
            if self._is_likely_ai_echo(text):
                logger.debug(f"🚫 跳过AI回答复述: {text}")
                return
                
            success = await self.openai_memory_manager.extract_and_store(text, context)
            
            if success:
                logger.debug(f"✅ 记忆处理成功: {text[:30]}...")
            
        except Exception as e:
            logger.debug(f"记忆处理异常: {e}")

    def _is_likely_ai_echo(self, text: str) -> bool:
        """判断是否是用户复述AI回答"""
        # 检查最近的AI回答
        recent_responses = getattr(self, 'recent_ai_responses', [])
        
        for ai_response in recent_responses[-3:]:  # 检查最近3个回答
            # 计算相似度
            similarity = self._text_similarity(text, ai_response)
            if similarity > 0.7:  # 70%相似度认为是复述
                return True
        
        # 检查典型的复述模式
        echo_patterns = [
            "我叫", "我是", "我的", "我在", "我住", "我工作",
            "我妈妈", "我爸爸", "我喜欢"
        ]
        
        # 如果包含多个自我描述词，可能是复述
        pattern_count = sum(1 for pattern in echo_patterns if pattern in text)
        if pattern_count >= 2 and len(text) < 50:
            return True
        
        return False

    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单文本相似度计算"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _should_respond(self, text):
        """判断是否需要回应"""
        questions = ['什么', '怎么', '为什么', '哪里', '谁', '如何', '吗', '呢', '？', '?']
        requests = ['帮我', '告诉我', '推荐', '建议']
        excludes = ['你好', '再见', '谢谢']
        
        has_question = any(q in text for q in questions)
        has_request = any(r in text for r in requests)
        is_excluded = any(e in text for e in excludes)
        
        return (has_question or has_request) and not is_excluded and len(text) > 3
    
    def _build_context(self):
        """构建上下文"""
        if not self.speech_segments:
            return ""
        recent = self.speech_segments[-3:]  # 最近3个片段
        return ' '.join([seg['text'] for seg in recent])
    
    async def _response_handler(self):
        """响应处理器"""
        logger.info("🔄 响应处理器启动")
        
        while self.running:
            try:
                response_data = await asyncio.wait_for(
                    self.response_queue.get(),
                    timeout=1.0
                )
                
                if not self.running:
                    break
                
                async with self.response_semaphore:
                    task_id = response_data['id']
                    task = asyncio.create_task(
                        self._generate_and_speak(response_data),
                        name=f"resp_{task_id}"
                    )
                    
                    self.active_responses[task_id] = task
                    logger.info(f"🎯 启动回答[{task_id}]")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"响应处理错误: {e}")
                await asyncio.sleep(0.5)
    
    async def _generate_and_speak(self, response_data):
        """生成并播放回答"""
        task_id = response_data['id']
        
        try:
            if not self.running:
                return
                
            final_text = response_data['final_text']
            context = response_data.get('text', '')
            prior_thought = response_data.get('prior_thought', {})
            
            logger.info(f"💭 生成回答[{task_id}]: {final_text}")
            
            # === 首先尝试OpenAI记忆回答 ===
            openai_response = ""
            used_memory_ids = []
            if self.memory_enabled and self.openai_memory_manager:
                try:
                    openai_response, used_memory_ids = await self.openai_memory_manager.smart_search_and_respond(
                        final_text, context
                    )
                except Exception as e:
                    logger.debug(f"OpenAI记忆回答失败: {e}")

            full_response = ""
            if openai_response and len(openai_response.strip()) > 5:
                logger.info(f"🧠 OpenAI记忆回答[{task_id}]: {openai_response}")
                full_response = openai_response
                
                # 立即记录使用了记忆的正面反馈
                if used_memory_ids:
                    for memory_id in used_memory_ids:
                        asyncio.create_task(
                            self.openai_memory_manager.importance_adjuster.track_user_feedback(
                                memory_id, True
                            )
                        )
                    logger.debug(f"📈 记录正面反馈: {len(used_memory_ids)} 条记忆")
            else:
                # 备用AI回答逻辑
                thought_hint = ""
                if prior_thought and prior_thought.get('thought'):
                    thought_hint = f" (预判:{prior_thought['thought']})"
                    
                prompt = f"""简洁回答：{final_text}{thought_hint}

    要求：1-2句话，实用直接："""
                
                try:
                    stream = await self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=80,
                        temperature=0.6,
                        stream=True
                    )
                    
                    async for chunk in stream:
                        if not self.running:
                            break
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        
                except Exception as e:
                    logger.error(f"OpenAI生成失败[{task_id}]: {e}")
                    full_response = "抱歉，暂时无法回答。"
            
            # 播放回答
            if full_response.strip() and self.running:
                logger.info(f"🎯 回答[{task_id}]: {full_response}")
                await self.tts_engine.speak_streaming(full_response, task_id)
            
            logger.info(f"✅ 回答完成[{task_id}]")
            

            
                    
        except Exception as e:
            if self.running:
                logger.error(f"生成回答失败[{task_id}]: {e}")
        finally:
            if task_id in self.active_responses:
                del self.active_responses[task_id]
        
    
    async def _cleanup_manager(self):
        """清理管理器"""
        while self.running:
            try:
                await asyncio.sleep(10)  # 每10秒清理一次
                current_time = time.time()
                
                # 清理过期的思考结果
                expired_thoughts = [
                    tid for tid, data in self.partial_thoughts.items()
                    if current_time - data['timestamp'] > 30
                ]
                
                for tid in expired_thoughts:
                    del self.partial_thoughts[tid]
                
                # 清理过期的思考时间
                expired_times = [
                    tid for tid, timestamp in self.last_thinking_time.items()
                    if current_time - timestamp > 60
                ]
                
                for tid in expired_times:
                    del self.last_thinking_time[tid]
                
                # 清理过期的对话片段
                if current_time - self.last_speech_time > 60:
                    self.speech_segments = []
                
                # === 新增：记忆系统清理 ===
                if self.memory_enabled and self.openai_memory_manager:
                    # 每小时清理一次旧记忆
                    if current_time % 3600 < 10:  # 1小时 = 3600秒
                        try:
                            await self.openai_memory_manager.cleanup_old_memories(days=30)
                        except Exception as e:
                            logger.debug(f"记忆清理失败: {e}")
                
                # 状态报告
                if len(self.active_responses) > 0:
                    logger.debug(f"📊 活跃回答: {len(self.active_responses)}")
                    
            except Exception as e:
                if self.running:
                    logger.error(f"清理管理错误: {e}")
    
    async def stop(self):
        """停止系统"""
        logger.info("🛑 停止系统...")
        self.running = False
        
        # 停止TTS
        await self.tts_engine.stop()
        
        # === 新增：关闭OpenAI记忆系统 ===
        if hasattr(self, 'openai_memory_manager') and self.openai_memory_manager:
            try:
                self.openai_memory_manager.close()
                logger.info("✅ OpenAI记忆系统已关闭")
            except Exception as e:
                logger.debug(f"关闭记忆系统失败: {e}")
        
        # 取消活跃任务
        for task_id, task in list(self.active_responses.items()):
            if not task.done():
                task.cancel()
        
        for task_id, task in list(self.thinking_tasks.items()):
            if not task.done():
                task.cancel()
        
        # 停止语音识别
        if self.recognizer:
            try:
                self.recognizer.stop_continuous_recognition()
                logger.info("✅ 语音识别已停止")
            except:
                pass
        
        logger.info("✅ 系统完全停止")

async def main():
    """主函数"""
    enhancer = FixedVoiceEnhancer()
    
    try:
        await enhancer.start()
    except KeyboardInterrupt:
        logger.info("👋 用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        if enhancer.running:
            await enhancer.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
    finally:
        print("✅ 程序已退出")