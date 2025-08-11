"""
配置文件 - 实时语音交互增强器
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@dataclass
class AudioConfig:
    """音频配置"""
    sample_rate: int = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
    chunk_size: int = int(os.getenv('AUDIO_CHUNK_SIZE', '1024'))
    channels: int = int(os.getenv('AUDIO_CHANNELS', '1'))
    format: int = int(os.getenv('AUDIO_FORMAT', '16'))
    
    # VAD 配置
    vad_mode: int = 3  # 0-3, 3最激进
    vad_frame_duration: int = 30  # ms
    
    # 语音检测阈值
    silence_threshold: float = 0.01
    speech_threshold: float = 0.3
    min_speech_duration: float = 0.5  # 秒
    max_silence_duration: float = 2.0  # 秒

@dataclass
class AzureConfig:
    """Azure Speech 配置"""
    speech_key: str = os.getenv('AZURE_SPEECH_KEY', '')
    speech_region: str = os.getenv('AZURE_SPEECH_REGION', 'eastus')
    
    # 语音识别配置
    language: str = 'zh-CN'
    recognition_mode: str = 'conversation'  # interactive, conversation, dictation
    
    # 说话人分离配置
    enable_speaker_diarization: bool = True
    max_speakers: int = 10

@dataclass
class OpenAIConfig:
    """OpenAI 配置"""
    api_key: str = os.getenv('OPENAI_API_KEY', '')
    model: str = 'gpt-3.5-turbo'
    max_tokens: int = int(os.getenv('MAX_RESPONSE_LENGTH', '200'))
    temperature: float = 0.7
    
    # 流式配置
    stream: bool = True
    stop_sequences: list = None

@dataclass
class TTSConfig:
    """TTS 配置"""
    engine: str = os.getenv('TTS_ENGINE', 'edge-tts')
    voice: str = os.getenv('TTS_VOICE', 'zh-CN-XiaoxiaoNeural')
    speed: float = float(os.getenv('TTS_SPEED', '1.0'))
    
    # 音频输出配置
    output_format: str = 'mp3'
    output_quality: str = 'high'

@dataclass
class ResponseConfig:
    """响应配置"""
    delay_ms: int = int(os.getenv('RESPONSE_DELAY_MS', '100'))
    context_window: int = int(os.getenv('CONTEXT_WINDOW_SIZE', '10'))
    
    # 触发条件
    min_confidence: float = 0.6
    require_question: bool = False  # 是否只对问题响应
    
    # 个性化
    default_personality: str = os.getenv('DEFAULT_PERSONALITY', 'casual_smart')
    user_profile_path: str = os.getenv('USER_PROFILE_PATH', 'data/personalities/user_profiles.json')
    # === 新增：智能记忆配置 ===
    smart_memory_enabled: bool = True
    memory_conflict_threshold: float = 0.8
    memory_quality_threshold: float = 0.6
    max_memory_age_days: int = 30
    auto_cleanup_enabled: bool = True
    intent_analysis_cache_size: int = 100
    conflict_resolution_strategy: str = "ai_assisted"  # "ai_assisted" 或 "rule_based"
@dataclass
class LogConfig:
    """日志配置"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    file_path: str = os.getenv('LOG_FILE', 'data/logs/voice_enhancer.log')
    
    # 日志格式
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    rotation: str = "10 MB"
    retention: str = "1 week"

class Config:
    """主配置类"""
    
    def __init__(self):
        self.audio = AudioConfig()
        self.azure = AzureConfig()
        self.openai = OpenAIConfig()
        self.tts = TTSConfig()
        self.response = ResponseConfig()
        self.log = LogConfig()
        
        # 验证必要配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置有效性"""
        if not self.azure.speech_key:
            raise ValueError("Azure Speech Key 未配置")
        
        if not self.openai.api_key:
            raise ValueError("OpenAI API Key 未配置")
        
        # 创建必要目录
        os.makedirs('data/personalities', exist_ok=True)
        os.makedirs('data/audio_cache', exist_ok=True)
        os.makedirs('data/logs', exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'audio': self.audio.__dict__,
            'azure': self.azure.__dict__,
            'openai': self.openai.__dict__,
            'tts': self.tts.__dict__,
            'response': self.response.__dict__,
            'log': self.log.__dict__
        }

# 全局配置实例
config = Config()