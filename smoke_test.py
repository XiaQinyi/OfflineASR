# -*- coding: utf-8 -*-
"""验证 Windows 本机能否加载 WeNet 模型并识别短音频。"""
from pathlib import Path
import sys
from app import MODEL_DIR, check_model_files, load_asr_model, make_wav_segments, decode_chunk
import tempfile

if len(sys.argv) != 2:
    print('用法: python smoke_test.py "C:\\path\\test.wav"')
    raise SystemExit(2)

check_model_files()
source = Path(sys.argv[1]).expanduser().resolve()
assert source.is_file(), f"音频不存在：{source}"
print(f"模型路径: {MODEL_DIR}")
model = load_asr_model()
print("模型加载成功")
with tempfile.TemporaryDirectory(prefix="asr_smoke_") as temp:
    clips = make_wav_segments(source, Path(temp))
    print("音频分段数量:", len(clips))
    for i, clip in enumerate(clips, 1):
        print(f"第{i}段：{decode_chunk(model, clip)}")
print("推理测试成功")
