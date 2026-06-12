#!/usr/bin/env python3
"""
누끼 툴 — OpenAI gpt-image-1 API로 N개 이미지 배경 제거
usage: python remove_bg.py image1.jpg image2.png ... [-o output_dir] [-j workers]
"""
import os
import sys
import io
import base64
import argparse
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from openai import OpenAI


def preprocess_for_api(im: Image.Image) -> tuple[bytes, dict]:
    """API 전송용 정사각형 PNG 생성. 원본 복원에 필요한 패딩 메타데이터도 반환."""
    orig_w, orig_h = im.size
    size = max(orig_w, orig_h)

    pad_x = (size - orig_w) // 2
    pad_y = (size - orig_h) // 2

    padded = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    padded.paste(im, (pad_x, pad_y))

    api_size = size
    if size > 4096:
        padded = padded.resize((4096, 4096), Image.LANCZOS)
        api_size = 4096

    buf = io.BytesIO()
    padded.save(buf, "PNG")

    meta = {
        "orig_w": orig_w, "orig_h": orig_h,
        "pad_x": pad_x, "pad_y": pad_y,
        "padded_size": size, "api_size": api_size,
    }
    return buf.getvalue(), meta


def extract_alpha_for_original(api_result: Image.Image, meta: dict) -> Image.Image:
    """API 결과(1024x1024)에서 알파 마스크만 뽑아 원본 해상도로 역변환."""
    # 1) API 결과 → 패딩 포함 정사각형 크기로 업스케일
    padded_size = meta["padded_size"]
    alpha = api_result.getchannel("A").resize(
        (padded_size, padded_size), Image.LANCZOS
    )

    # 2) 패딩 제거 → 원본 영역만 크롭
    orig_w, orig_h = meta["orig_w"], meta["orig_h"]
    pad_x, pad_y = meta["pad_x"], meta["pad_y"]
    alpha_cropped = alpha.crop((pad_x, pad_y, pad_x + orig_w, pad_y + orig_h))

    return alpha_cropped


def remove_background(src_path: Path, dst_path: Path, client: OpenAI) -> str:
    """단일 이미지 누끼. 원본 해상도/품질 유지, 배경만 투명화."""
    # 원본을 원본 그대로 보관
    original = Image.open(src_path).convert("RGBA")

    png_bytes, meta = preprocess_for_api(original)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            response = client.images.edit(
                model="gpt-image-1",
                image=f,
                prompt=(
                    "Remove the background completely. "
                    "Make the background fully transparent. "
                    "Keep only the main subject with clean edges."
                ),
                n=1,
                size="1024x1024",
            )
    finally:
        os.unlink(tmp_path)

    item = response.data[0]
    if hasattr(item, "b64_json") and item.b64_json:
        img_bytes = base64.b64decode(item.b64_json)
    elif hasattr(item, "url") and item.url:
        import urllib.request
        with urllib.request.urlopen(item.url) as r:
            img_bytes = r.read()
    else:
        raise ValueError("API 응답에 이미지 데이터 없음")

    api_result = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # API 알파 마스크를 원본 해상도로 역변환 → 원본에 씌우기
    alpha_mask = extract_alpha_for_original(api_result, meta)
    original.putalpha(alpha_mask)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    original.save(dst_path, "PNG")
    return "gpt-image-1"


def main():
    parser = argparse.ArgumentParser(
        description="누끼 툴 — OpenAI gpt-image-1로 N개 이미지 배경 제거"
    )
    parser.add_argument("images", nargs="+", help="이미지 파일 경로")
    parser.add_argument(
        "-o", "--output", default="output_nuki", help="출력 폴더 (기본값: output_nuki)"
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=4, help="병렬 처리 수 (기본값: 4)"
    )
    parser.add_argument(
        "--suffix", default="_nuki", help="출력 파일명 접미사 (기본값: _nuki)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 없습니다.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for raw in args.images:
        src = Path(raw)
        if not src.exists():
            print(f"  [건너뜀] 파일 없음: {src}")
            continue
        dst = out_dir / (src.stem + args.suffix + ".png")
        tasks.append((src, dst))

    total = len(tasks)
    if total == 0:
        print("처리할 이미지가 없습니다.")
        return

    print(f"\n총 {total}개 이미지 누끼 시작 (model=gpt-image-1, workers={args.workers})\n")

    def process(item):
        src, dst = item
        try:
            method = remove_background(src, dst, client)
            return src.name, dst, method, None
        except Exception as e:
            return src.name, dst, None, str(e)

    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            name, dst, method, err = fut.result()
            if err:
                print(f"  [{i}/{total}] ✗ {name}  오류: {err}")
            else:
                print(f"  [{i}/{total}] ✓ {name}  →  {dst.name}")
                success += 1

    print(f"\n완료: {success}/{total}개 성공  →  {out_dir.resolve()}")

    if sys.platform == "darwin" and success > 0:
        os.system(f'open "{out_dir.resolve()}"')


if __name__ == "__main__":
    main()
