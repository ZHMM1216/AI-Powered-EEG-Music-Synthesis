#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
请在这里填写您的 API 密钥和配置
"""

# ==========================================
# Suno API 配置
# ==========================================

# Suno API 密钥
# 获取地址: https://sunoapi.org/api-key
SUNO_API_KEY = "your-token"

# API 基础 URL
SUNO_API_BASE_URL = "https://api.sunoapi.org"

# 回调 URL（Suno API 必填字段，但可留空）
# 留空时将自动使用默认占位URL: http://localhost/callback （使用轮询模式查询结果）
# 如果你有真实的回调服务器，请填写完整的 HTTP(S) URL
SUNO_CALLBACK_URL = ""  # 留空时使用默认占位URL

# ==========================================
# 其他配置
# ==========================================

# 音乐保存文件夹
GENERATED_MUSIC_FOLDER = "./generated_music"

# 模型文件路径
MODEL_PATH = "./best_model.pth"

# 服务器配置
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SERVER_DEBUG = False  # 关闭调试模式以正常显示日志
