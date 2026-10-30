# -*- coding: utf-8 -*-
"""Windows 离线语音识别图形界面（使用本地 WeNet Conformer 模型）。

先在 Windows CPU 环境安装兼容的 WeNet 及 PyTorch，再运行本程序；
模型文件需放在与 app.py 同目录的 model/ 下。
"""
from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def program_dir() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


BASE_DIR = program_dir()
MODEL_DIR = BASE_DIR / "model"
REQUIRED_MODEL_FILES = ("final.pt", "train.yaml", "units.txt")


def check_model_files() -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少模型文件：" + "、".join(missing) + "。请将 AutoDL 的 windows_model.zip 解压到当前项目目录，确保存在 model/。"
        )


def make_wav_segments(audio_path: Path, folder: Path, seconds: int = 20) -> list[Path]:
    """使用 FFmpeg 统一为 16kHz/mono PCM WAV，并以固定时长分段。

    警告：固定切分不具备 VAD；切分点可能截断词语，长录音精度可能下降。
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = folder / "chunk_%05d.wav"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(audio_path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-f", "segment", "-segment_time", str(seconds),
        "-reset_timestamps", "1", str(pattern),
    ]
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(command, capture_output=True, text=True, creationflags=flags)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg 转换失败：" + (result.stderr.strip() or "未知错误"))
    segments = sorted(folder.glob("chunk_*.wav"))
    if not segments:
        raise RuntimeError("未提取到音频，请确认文件包含可读取的音轨。")
    return segments


def load_asr_model():
    import torch
    import wenet

    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model = wenet.load_model(str(MODEL_DIR), device="cpu")
    model.eval()
    return model


def decode_chunk(model, wav: Path) -> str:
    """与 AISHELL-1 Attention Rescoring 评估尽量保持一致的解码参数。

    对应原评估：beam_size=10, ctc_weight=0.3, reverse_weight=0.5。
    """
    import torch

    device = next(model.parameters()).device
    with torch.inference_mode():
        feat = model.compute_feature(str(wav)).to(device)
        lengths = torch.tensor([feat.size(0)], dtype=torch.long, device=device)
        result = model.decode(
            ["attention_rescoring"], feat.unsqueeze(0), lengths,
            beam_size=10, ctc_weight=0.3, reverse_weight=0.5,
        )["attention_rescoring"][0]
        text = model.tokenizer.detokenize(result.tokens)[0]
    return text.strip()


class ASRWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("离线语音识别系统 - AISHELL-1 Conformer")
        self.root.geometry("860x590")
        self.root.minsize(690, 440)
        self.root.configure(bg="#f4f6fb")

        self.file_path = tk.StringVar()
        self.status = tk.StringVar(value="就绪：请选择一个 MP3 或 WAV 文件")
        self.events: queue.Queue[tuple] = queue.Queue()
        self.running = False
        self.text_result = ""
        self.model = None

        tk.Label(root, text="离线语音识别系统", font=("Microsoft YaHei UI", 21, "bold"),
                 bg="#f4f6fb", fg="#172554").pack(pady=(21, 5))
        tk.Label(root, text="WeNet Conformer · Attention Rescoring（Beam 10）· 完全本地推理",
                 font=("Microsoft YaHei UI", 10), bg="#f4f6fb", fg="#58677d").pack(pady=(0, 18))

        top = tk.Frame(root, bg="#f4f6fb")
        top.pack(fill="x", padx=27)
        ttk.Entry(top, textvariable=self.file_path).pack(side="left", fill="x", expand=True, ipady=5)
        self.browse_btn = ttk.Button(top, text="选择音频", command=self.choose_file)
        self.browse_btn.pack(side="left", padx=(9, 0))

        actions = tk.Frame(root, bg="#f4f6fb")
        actions.pack(fill="x", padx=27, pady=15)
        self.run_btn = ttk.Button(actions, text="开始识别", command=self.start)
        self.run_btn.pack(side="left")
        self.save_btn = ttk.Button(actions, text="保存为 TXT", command=self.save, state="disabled")
        self.save_btn.pack(side="left", padx=9)
        self.clear_btn = ttk.Button(actions, text="清空结果", command=self.clear)
        self.clear_btn.pack(side="left")

        self.progress = ttk.Progressbar(root, mode="indeterminate")
        self.progress.pack(fill="x", padx=27, pady=(0, 9))
        tk.Label(root, textvariable=self.status, anchor="w", bg="#f4f6fb", fg="#475569",
                 font=("Microsoft YaHei UI", 9)).pack(fill="x", padx=27, pady=(0, 9))
        frame = tk.Frame(root, bg="#fff", bd=1, relief="solid")
        frame.pack(fill="both", expand=True, padx=27, pady=(0, 12))
        self.textbox = tk.Text(frame, wrap="word", font=("Microsoft YaHei UI", 12),
                               relief="flat", padx=13, pady=12, spacing2=4)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.textbox.yview)
        self.textbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.textbox.pack(side="left", fill="both", expand=True)

        tk.Label(root, text="提示：长音频采用约 20 秒固定分段，可能在切分处影响识别；首次 CPU 加载模型需要一些时间。",
                 bg="#f4f6fb", fg="#64748b", font=("Microsoft YaHei UI", 9)).pack(pady=(0, 14))
        self.root.after(120, self.pump_events)

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择音频文件", filetypes=[("音频文件", "*.wav *.mp3"), ("全部文件", "*.*")]
        )
        if path:
            self.file_path.set(path)

    def start(self) -> None:
        if self.running:
            return
        audio = Path(self.file_path.get().strip().strip('"'))
        if not audio.is_file():
            messagebox.showwarning("缺少文件", "请先选择一个存在的音频文件。")
            return
        if audio.suffix.lower() not in (".wav", ".mp3"):
            messagebox.showwarning("格式不支持", "当前仅支持 WAV 和 MP3。")
            return
        try:
            check_model_files()
        except Exception as exc:
            messagebox.showerror("模型文件缺失", str(exc))
            return
        self.running = True
        self.text_result = ""
        self.textbox.delete("1.0", "end")
        self.run_btn.configure(state="disabled")
        self.browse_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.progress.start(9)
        self.status.set("正在初始化识别...")
        threading.Thread(target=self.work, args=(audio,), daemon=True).start()

    def work(self, audio: Path) -> None:
        try:
            if self.model is None:
                self.events.put(("status", "正在从本地加载 Conformer 模型（CPU）..."))
                self.model = load_asr_model()
            with tempfile.TemporaryDirectory(prefix="offline_asr_") as temp:
                folder = Path(temp)
                self.events.put(("status", "正在转换音频格式并切分..."))
                segments = make_wav_segments(audio, folder)
                texts: list[str] = []
                for i, chunk in enumerate(segments, 1):
                    self.events.put(("status", f"正在识别第 {i}/{len(segments)} 段..."))
                    text = decode_chunk(self.model, chunk)
                    if text:
                        texts.append(text)
                    self.events.put(("partial", "\n".join(texts)))
                self.events.put(("done", "\n".join(texts)))
        except Exception as exc:
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))

    def pump_events(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "status":
                    self.status.set(value)
                elif event == "partial":
                    self.textbox.delete("1.0", "end")
                    self.textbox.insert("1.0", value)
                    self.textbox.see("end")
                elif event == "done":
                    self.text_result = value
                    self.textbox.delete("1.0", "end")
                    self.textbox.insert("1.0", value)
                    self.status.set("识别完成，可保存 TXT" if value else "识别已结束，但没有识别出文字")
                    self.end_run()
                elif event == "error":
                    self.status.set("识别失败，请查看错误提示")
                    self.end_run()
                    messagebox.showerror("识别失败", value)
        except queue.Empty:
            pass
        self.root.after(120, self.pump_events)

    def end_run(self) -> None:
        self.running = False
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")
        if self.text_result.strip():
            self.save_btn.configure(state="normal")

    def save(self) -> None:
        text = self.textbox.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("没有文本", "请先完成一次识别。")
            return
        src = Path(self.file_path.get().strip().strip('"'))
        target = filedialog.asksaveasfilename(
            title="保存识别结果", defaultextension=".txt",
            initialfile=(src.stem or "识别结果") + ".txt",
            filetypes=[("文本文件", "*.txt")]
        )
        if target:
            Path(target).write_text(text + "\n", encoding="utf-8-sig")
            self.status.set(f"已保存 TXT：{target}")
            messagebox.showinfo("保存成功", "识别结果已保存到：\n" + target)

    def clear(self) -> None:
        if self.running:
            return
        self.text_result = ""
        self.textbox.delete("1.0", "end")
        self.save_btn.configure(state="disabled")
        self.status.set("就绪")


if __name__ == "__main__":
    root = tk.Tk()
    ASRWindow(root)
    root.mainloop()
