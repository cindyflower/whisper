"""
可選的說話者分離（與本機 OpenAI Whisper 分開）。

需：
  pip install pyannote.audio
並至 Hugging Face 接受
  https://huggingface.co/pyannote/segmentation-3.0
  https://huggingface.co/pyannote/speaker-diarization-3.1
之使用條款，再設定：
  export HF_TOKEN=你的_hf_token
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch

DiarSeg = Tuple[float, float, str]  # start_s, end_s, speaker_id


def get_hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def run_speaker_diarization(audio_path: str, hf_token: str) -> List[DiarSeg]:
    """
    以 pyannote 取得每段的 (開始秒, 結束秒, 說話者 id)。
    失敗時拋出例外，由 UI 轉成可讀訊息。
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "未安裝 pyannote。請在虛擬環境內執行： pip install pyannote.audio"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    pipeline.to(device)

    diarization = pipeline(audio_path)
    out: List[DiarSeg] = []
    for turn, _track, label in diarization.itertracks(yield_label=True):
        out.append((float(turn.start), float(turn.end), str(label)))
    out.sort(key=lambda x: x[0])
    if not out:
        raise RuntimeError("說話者模型未產生任何片段（可能音檔過短或無人聲）。")
    return out
