import os
import sys


def _repo_id() -> str:
    return "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"


def _workspace_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _target_dir() -> str:
    # WhisperX/Transformers 读取对齐模型时更依赖 HuggingFace cache 结构（models--.../snapshots/...）
    # 因此这里把 repo 下载进 cache_dir=models/WhisperX
    return os.path.join(_workspace_root(), "models", "WhisperX")


def main() -> int:
    repo_id = _repo_id()
    target_dir = _target_dir()
    os.makedirs(target_dir, exist_ok=True)

    print(f"[INFO] repo_id: {repo_id}")
    print(f"[INFO] target_dir: {target_dir}")

    # 1) Prefer Hugging Face Hub cache_dir download (WhisperX-compatible).
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        print("[INFO] Using huggingface_hub.snapshot_download ...")
        out = snapshot_download(
            repo_id=repo_id,
            cache_dir=target_dir,
            local_dir_use_symlinks=False,  # windows-friendly, also fine on linux
            resume_download=True,
        )
        print(f"[OK] Downloaded (huggingface_hub cache_dir) to: {out}")
        return 0
    except Exception as e:
        print(f"[WARN] huggingface_hub snapshot_download failed, fallback to modelscope local_dir: {e}")

    # 2) Fallback to ModelScope local_dir download (may not match HF cache structure).
    try:
        from modelscope import snapshot_download  # type: ignore

        local_dir = os.path.join(target_dir, "modelscope--" + repo_id.replace("/", "--"))
        os.makedirs(local_dir, exist_ok=True)
        print("[INFO] Using modelscope.snapshot_download ...")
        out = snapshot_download(repo_id, local_dir=local_dir)
        print(f"[OK] Downloaded (modelscope) to: {out}")
        print("[WARN] ModelScope 下载不一定满足 WhisperX 的 HF cache 结构；若仍报 preprocessor_config.json 缺失，请优先使用 huggingface_hub。")
        return 0
    except Exception as e:
        print(f"[FAIL] Download failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

