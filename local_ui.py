"""
本機專用 Whisper 介面：只綁定 127.0.0.1，不建立公開連結。
音檔在記憶體／暫存目錄處理，不傳到任何遠端服務（關掉瀏覽器後可依需求自行刪除暫存）。
"""

from __future__ import annotations

import os

# 必須在 import gradio/httpx 之前設定，否則本機自檢會走系統代理而失敗
def _merge_no_proxy_for_localhost() -> None:
    """避免系統的 HTTP/HTTPS 代理導致 Gradio 內部 httpx 無法連 127.0.0.1，進而啟動失敗。"""
    # 勿加入 [::1]（含方括號），部分 httpx 版本會把 NO_PROXY 誤解析成 URL 而崩潰
    extra = ("127.0.0.1", "localhost", "0.0.0.0")
    for var in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(var, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for h in extra:
            if h not in parts:
                parts.append(h)
        os.environ[var] = ",".join(parts)


_merge_no_proxy_for_localhost()
if os.environ.get("GRADIO_SSR_MODE", "").lower() in ("1", "true", "yes"):
    os.environ["GRADIO_SSR_MODE"] = "false"

import traceback
import webbrowser
from collections.abc import Iterator
from typing import Any, List, Optional, Tuple

import gradio as gr
import torch
import whisper
from whisper.tokenizer import LANGUAGES
from whisper.utils import format_timestamp

import whisper_diarize
from whisper_diarize import DiarSeg

_models: dict[str, Any] = {}


def _language_choices() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = [("自動偵測", "auto")]
    for code, name in sorted(LANGUAGES.items(), key=lambda x: x[1]):
        out.append((f"{name} ({code})", code))
    return out


def _resolve_device(choice: str) -> Optional[str]:
    if choice == "default":
        return None
    if choice == "mps" and torch.backends.mps.is_available():
        return "mps"
    if choice == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if choice == "cuda" and torch.cuda.is_available():
        return "cuda"
    if choice == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if choice == "cpu":
        return "cpu"
    return None


def get_model(name: str, device_choice: str) -> Any:
    dev = _resolve_device(device_choice)
    key = f"{name!s}|{dev!r}"
    if key not in _models:
        _models[key] = whisper.load_model(name, device=dev)
    return _models[key]


def _as_local_path(f: Any) -> Optional[str]:
    """Gradio File 常回傳 NamedString（子類別 str）；統一成本機路徑。"""
    if f is None:
        return None
    p = str(f).strip()
    if not p or not os.path.isfile(p):
        return None
    return p


def _pick_input_path(
    file_upload: Any,
    audio: Any,
) -> Optional[str]:
    """優先使用檔案上傳；否則用麥克風錄音的暫存路徑。"""
    p1 = _as_local_path(file_upload)
    if p1:
        return p1
    return _as_local_path(audio)


def _assign_speaker_for_segment(
    s0: float, s1: float, diar: list[DiarSeg]
) -> Optional[str]:
    mid = 0.5 * (s0 + s1)
    for a, b, spk in diar:
        if a <= mid <= b:
            return spk
    best_spk, best_o = None, 0.0
    for a, b, spk in diar:
        o = max(0.0, min(s1, b) - max(s0, a))
        if o > best_o:
            best_o, best_spk = o, spk
    return best_spk


def _speaker_display_names(diar: list[DiarSeg]) -> dict[str, str]:
    order: list[str] = []
    for *_, spk in diar:
        if spk not in order:
            order.append(spk)
    return {s: f"說話者{i + 1}" for i, s in enumerate(order)}


def _format_transcript(
    result: dict,
    diar: Optional[list[DiarSeg]] = None,
) -> str:
    """
    依 Whisper 的 segments 分段；若提供 diar，則併上說話者並合併連續同說話者。
    """
    segs = result.get("segments") or []
    if not segs:
        return (result.get("text") or "").strip()

    spk_map: dict[str, str] = {}
    if diar:
        spk_map = _speaker_display_names(diar)

    rows: list[tuple[str, float, float, str]] = []
    for seg in segs:
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        s0, s1 = float(seg["start"]), float(seg["end"])
        if diar:
            raw = _assign_speaker_for_segment(s0, s1, diar)
            dsp = spk_map.get(raw, raw or "（未知）")
        else:
            dsp = ""
        rows.append((dsp, s0, s1, t))

    if not diar:
        parts: list[str] = []
        for _dsp, s0, s1, t in rows:
            t0 = format_timestamp(s0, always_include_hours=True)
            t1 = format_timestamp(s1, always_include_hours=True)
            parts.append(f"[{t0} – {t1}]\n{t}")
        return "\n\n".join(parts)

    merged: list[tuple[str, float, float, str]] = []
    for dsp, s0, s1, t in rows:
        if merged and merged[-1][0] == dsp:
            p = merged[-1]
            merged[-1] = (dsp, p[1], s1, p[3] + " " + t)
        else:
            merged.append((dsp, s0, s1, t))

    parts = []
    for dsp, s0, s1, t in merged:
        t0 = format_timestamp(s0, always_include_hours=True)
        t1 = format_timestamp(s1, always_include_hours=True)
        parts.append(f"[{t0} – {t1}] {dsp}\n{t}")
    return "\n\n".join(parts)


def transcribe(
    file_upload: Any,
    audio: Any,
    use_speakers: bool,
    model_name: str,
    language: str,
    task: str,
    device_choice: str,
) -> Iterator[Tuple[str, str]]:
    print("[whisper UI] 收到轉寫請求，開始檢查輸入…", flush=True)
    yield ("狀態：已收到請求，正在檢查音檔…", "")
    path = _pick_input_path(file_upload, audio)
    if not path:
        yield (
            "狀態：未拿到音檔。請在「上傳音檔」選擇檔案，或於下方錄音後再按「開始轉寫」。",
            "",
        )
        return
    if not os.path.isfile(path):
        yield (f"狀態：找不到音檔 {path!r}。", "")
        return

    opts: dict[str, Any] = {"task": task}
    if language and language != "auto":
        opts["language"] = language

    print(f"[whisper UI] 開始轉寫: path={path!r}, model={model_name!r}", flush=True)
    yield (
        "狀態：已收到音檔。正在載入模型（**首次使用該模型**時會下載權重，可能需數分鐘）…",
        "",
    )

    try:
        model = get_model(model_name, device_choice)
    except Exception:
        err = f"載入模型失敗：\n{traceback.format_exc()}"
        print(err, flush=True)
        yield ("狀態：錯誤（見下方）", err)
        return

    yield (
        "狀態：模型已就緒，正在轉寫中（**長音檔**可能需幾分鐘，畫面須稍候，勿關閉分頁）…",
        "",
    )
    try:
        result = model.transcribe(path, **opts)
    except Exception:
        err = f"轉寫失敗：\n{traceback.format_exc()}"
        print(err, flush=True)
        yield ("狀態：錯誤（見下方）", err)
        return

    if not (result.get("text") or "").strip() and not (result.get("segments")):
        yield (
            "狀態：完成，但未辨識到文字。可改選較大模型、指定語言，或檢查音檔。",
            "（空結果）",
        )
        return

    diar: Optional[list[DiarSeg]] = None
    if use_speakers:
        tok = whisper_diarize.get_hf_token()
        if not tok:
            yield (
                "狀態：已勾選「說話者分離」但未偵測到 HF token。"
                "請在終端執行 `export HF_TOKEN=你的token` 後重啟（token 在 huggingface.co 設定內可建立），"
                "並先至 Hugging Face 接受 pyannote 相關模型使用條款。下方改為**僅依時間分段**。",
                _format_transcript(result, diar=None),
            )
            return
        else:
            yield (
                "狀態：轉寫完成。正在以 pyannote 做說話者分離（**額外需一些時間**；若從未安裝則需先執行 "
                "`pip install pyannote.audio`）…",
                _format_transcript(result, diar=None),
            )
            try:
                diar = whisper_diarize.run_speaker_diarization(path, tok)
            except Exception as e:
                err = f"說話者分離失敗，已改為僅時間分段。詳情：\n{e!s}\n{traceback.format_exc()}"
                print(err, flush=True)
                yield ("狀態：完成（說話者分離失敗，見下方或終端訊息）。", _format_transcript(result, diar=None))
                return
    out_text = _format_transcript(result, diar=diar)
    note = "狀態：完成。" + (
        " 已盡力標示說話者（以語音特徵分群，**不保證與實際人名一致**）。"
        if diar
        else " 已依**時間**分段，方便閱讀（未做說話者模型）。"
    )
    yield (note, out_text)


def build_ui() -> gr.Blocks:
    models = whisper.available_models()
    default_model = "turbo" if "turbo" in models else models[0]

    with gr.Blocks(title="Whisper 本機轉寫") as demo:
        gr.Markdown(
            "### Whisper 本機轉寫\n"
            "- **只在本機瀏覽器開啟**（`127.0.0.1`），不產生公開分享連結。\n"
            "- 音訊在這台電腦上轉寫，**不會上傳到 OpenAI 或第三方雲端**；第一次使用某個模型時仍會從 "
            "OpenAI 的網址**下載權重**到本機快取 `~/.cache/whisper`（與指令列使用 Whisper 相同）。\n"
            "- **翻譯成英文**建議用 `medium` / `large` 等；`turbo` 不適合做翻譯任務。\n"
            "- **說話者分離**使用 pyannote（與人名／角色無關，僅依聲紋分「說話者1、2…」），需另安裝並設定 HF，見勾選方塊說明。",
        )
        with gr.Row():
            file_in = gr.File(
                file_count="single",
                type="filepath",
                label="上傳音檔（建議，較穩定；不限制副檔名，靠 ffmpeg/Whisper 解碼）",
            )
        with gr.Row():
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="或：麥克風錄音（上傳與擇一即可）",
            )
        with gr.Row():
            model_name = gr.Dropdown(
                choices=models,
                value=default_model,
                label="模型",
            )
            language = gr.Dropdown(
                choices=_language_choices(),
                value="auto",
                label="語言",
            )
            task = gr.Radio(
                choices=["transcribe", "translate"],
                value="transcribe",
                label="任務",
            )
        device_choice = gr.Radio(
            choices=[
                ("依套件預設（CUDA 或 CPU）", "default"),
                ("Apple GPU (MPS)", "mps"),
                ("CPU", "cpu"),
                ("NVIDIA (CUDA)", "cuda"),
            ],
            value="default",
            label="裝置",
        )
        use_speakers = gr.Checkbox(
            value=False,
            label="嘗試標示不同說話者（需 pip install pyannote.audio 與 HF token；較慢）",
            info="預設仍會用 Whisper **按時間分行**。勾選此項才會在終端有 HF 的前提下跑 pyannote 分聲。",
        )
        run_btn = gr.Button("開始轉寫", variant="primary")
        status = gr.Markdown("狀態：待機。")
        # Gradio 6：複製按鈕改由 Textbox 工具列提供（需 show_label=True）
        out = gr.Textbox(
            label="轉寫文字（已分段；有說話者時會併在時間行）",
            lines=20,
            max_lines=40,
            show_label=True,
            buttons=["copy"],
        )

        run_btn.click(
            fn=lambda: ("狀態：已送出，等待後端排程…", ""),
            outputs=[status, out],
            queue=False,
        ).then(
            fn=transcribe,
            inputs=[file_in, audio, use_speakers, model_name, language, task, device_choice],
            outputs=[status, out],
            show_progress="full",
        )
    return demo


def main() -> None:
    _merge_no_proxy_for_localhost()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(
        "\n"
        "==================================================\n"
        "  Whisper 本機介面：請保持此終端機視窗開啟；停止按 Ctrl+C\n"
        f"  預設埠：{port}（可設環境變數 GRADIO_SERVER_PORT 改埠）\n"
        "  請用一般瀏覽器（Safari/Chrome）開啟，勿用需代理的僅內部網址。\n"
        "  若出現「無法連線」：請以 http:// 開啟、不要用 https；\n"
        "  並在終端執行一次：export NO_PROXY=127.0.0.1,localhost 後再執行本程式。\n"
        "==================================================\n",
        flush=True,
    )
    demo = build_ui()
    demo.queue()
    try:
        _app, local_url, _share = demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True,
            theme=gr.themes.Soft(),
            ssr_mode=False,
            strict_cors=False,
            inbrowser=True,
        )
    except ValueError as e:
        if "localhost" in str(e).lower() or "share" in str(e).lower():
            print(
                "\n[提示] 本機迴路可能被代理攔下。可在終端執行後再重試：\n"
                "  export NO_PROXY=127.0.0.1,localhost,0.0.0.0\n"
                "  export no_proxy=127.0.0.1,localhost,0.0.0.0\n"
                f"  （原始訊息：{e!s}）\n",
                flush=True,
            )
        raise
    except Exception as e:
        if "startup-events" in str(e) or "failed" in str(e).lower():
            print(
                "\n[提示] Gradio 無法完成本機自檢，常見於代理／VPN。請關閉系統「自動代理」"
                "或手動加入 NO_PROXY 後重試。原始錯誤如下。\n",
                flush=True,
            )
        raise
    if local_url:
        clean = local_url.rstrip("/")
        print(f"\n  ** 請在瀏覽器使用此完整網址： {clean} **\n", flush=True)
        # 雙重保險：明確以 IPv4 開分頁（略過僅有 localhost / IPv6 的差異）
        if "127.0.0.1" in clean or "localhost" in clean:
            try:
                webbrowser.open(clean)
            except OSError:
                pass


if __name__ == "__main__":
    main()
