#!/usr/bin/env python3
import os
import sys
import base64
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI

def generate_images(prompt: str, count: int = 5, model: str = "gpt-image-1", size: str = "1024x1024"):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"\n모델: {model}")
    print(f"프롬프트: {prompt}")
    print(f"생성 중... ({count}개 동시 요청)\n")

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=count,
        size=size,
        response_format="b64_json",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, image_data in enumerate(response.data, 1):
        img_bytes = base64.b64decode(image_data.b64_json)
        path = output_dir / f"image_{i}.png"
        path.write_bytes(img_bytes)
        saved_paths.append(path)
        print(f"  [{i}/{count}] 저장 완료: {path}")

    print(f"\n완료! 저장 위치: {output_dir.resolve()}")

    # macOS에서 폴더 자동 오픈
    if sys.platform == "darwin":
        os.system(f'open "{output_dir.resolve()}"')

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="GPT 이미지 5개 동시 생성 툴")
    parser.add_argument("prompt", nargs="?", help="이미지 생성 프롬프트")
    parser.add_argument("-n", "--count", type=int, default=5, help="생성 개수 (기본값: 5)")
    parser.add_argument("-m", "--model", default="gpt-image-1", help="모델 (기본값: gpt-image-1)")
    parser.add_argument(
        "-s", "--size",
        default="1024x1024",
        choices=["1024x1024", "1536x1024", "1024x1536"],
        help="이미지 크기 (기본값: 1024x1024)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    prompt = args.prompt
    if not prompt:
        prompt = input("프롬프트 입력: ").strip()
        if not prompt:
            print("프롬프트를 입력해주세요.")
            sys.exit(1)

    generate_images(prompt, count=args.count, model=args.model, size=args.size)


if __name__ == "__main__":
    main()
