# Windows 离线语音识别软件——启动项目

> 当前状态：**源代码模板 / 待在你的 Windows 环境实际验证**，并不是已经验证可直接运行的 EXE。模型权重需从 AutoDL 的 `windows_model.zip` 获取。

## 0. 先在 AutoDL 记录与你训练一致的 WeNet 版本

```bash
cd ~/autodl-tmp/speech_project/wenet
git rev-parse HEAD
# 如果仓库没有未提交改动，可选：把训练用源码另存，避免 Windows 装到不兼容的新版本
git archive --format=zip --output=~/autodl-tmp/speech_project/wenet_source.zip HEAD
```

将 `windows_model.zip` 和 `wenet_source.zip` 从 AutoDL 下载到 Windows。

## 1. Windows 项目目录

解压此启动项目到 `D:\OfflineASR`，再把 AutoDL 的 `windows_model.zip` **解压到这个目录**，确保：

```
D:\OfflineASR\
├── app.py
├── smoke_test.py
├── build_exe.bat
└── model\
    ├── final.pt
    ├── train.yaml
    ├── units.txt
    ├── lang_char.txt
    └── global_cmvn
```

不要把 `windows_model.zip` 解压后的 `model` 再套一层 `windows_delivery` 目录。

## 2. Windows 先验证 Python 推理（联网安装，之后离线使用）

推荐 Windows 10/11 x64、Python 3.12 x64，命令在 **PowerShell** 中执行。训练环境为 Python 3.12、PyTorch 2.8.0；尽量先匹配这个组合。需要安装 Python 时，从 python.org 取得安装程序。

```powershell
cd D:\OfflineASR
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install "setuptools==80.9.0" wheel
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install pyyaml sentencepiece librosa jieba langid requests tqdm imageio-ffmpeg
.\.venv\Scripts\python.exe -m pip install --no-build-isolation "openai-whisper==20231117" --no-deps
.\.venv\Scripts\python.exe -m pip install tiktoken more-itertools
```

接下来解压 `wenet_source.zip` 到 `D:\OfflineASR\wenet_source`（其根目录内应有 `setup.py` 和 `wenet` 文件夹），安装与训练一致的源码：

```powershell
cd D:\OfflineASR\wenet_source
..\.venv\Scripts\python.exe -m pip install -e . --no-deps
cd D:\OfflineASR
.\.venv\Scripts\python.exe -c "import wenet, torch; print('wenet OK', torch.__version__)"
```

**注意：WeNet 及 PyTorch 的 Windows 兼容性尚未在这里实际测试。** 如果出现 `ModuleNotFoundError` / 动态库异常，先解决依赖，别急着打包。不能保证 Linux 环境里所有 WeNet 依赖都可直接安装到 Windows。

## 3. 短音频冒烟测试

先用短 WAV 或 MP3：

```powershell
cd D:\OfflineASR
.\.venv\Scripts\python.exe smoke_test.py "D:\voice\test.wav"
```

成功输出中文后，启动图形界面：

```powershell
.\.venv\Scripts\python.exe app.py
```

界面支持选择 MP3/WAV、开始识别、查看文本、保存 TXT。`imageio-ffmpeg` 自带 FFmpeg 二进制可供离线转换，无需通过互联网识别。

### 识别方式

使用训练得出的 **Attention Rescoring**：`beam_size=10`、`ctc_weight=0.3`、`reverse_weight=0.5`。长音频按 20 秒分段识别（不是端到端长音频/VAD；切分处可能有识别错误），适合先完成课堂作业演示。

## 4. Windows 打包 EXE（必须先通过 Windows 推理测试）

打包前关闭正在运行的 app，执行：

```powershell
cd D:\OfflineASR
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\build_exe.bat
```

预计得到：

```
dist\OfflineASR\OfflineASR.exe
```

保留 `dist\OfflineASR` **整个文件夹** 一起交付，不能只拷贝单独的 EXE。`torch` + `wenet` 的依赖较复杂，PyInstaller 可能需要继续补 DLL 或 hidden-import；这份脚本是首次尝试，不保证一次通过。最终必须在一台无 Python 的 Windows 电脑上，断网测试 WAV/MP3 识别与 TXT 导出。

## 5. 老师的对比实验结果

AISHELL-1 test（7176 条）：

- CTC Greedy Search CER：5.90%
- Attention Rescoring CER：5.27%
- 绝对降低 0.63 个百分点，相对错误率降低约 10.68%

注意：这属于 **解码方法对比**，不能直接当作“微调前后对比”。若老师要求微调实验，需另行补做。
