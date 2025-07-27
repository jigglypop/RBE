#!/usr/bin/env python3
"""
KoGPT-2 토크나이저 다운로드 스크립트
"""

import os
import json
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

def download_kogpt2_tokenizer():
    """HuggingFace에서 KoGPT-2 토크나이저를 다운로드"""
    
    print("🔽 KoGPT-2 토크나이저 다운로드 중...")
    
    # 디렉토리 생성
    os.makedirs("models/skt-kogpt2-base-v2", exist_ok=True)
    
    # HuggingFace에서 KoGPT-2 토크나이저 다운로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
    
    # 토크나이저 파일 저장
    tokenizer.save_pretrained("models/skt-kogpt2-base-v2")
    
    print("✅ KoGPT-2 토크나이저 저장 완료!")
    
    # 테스트
    print("\n🧪 토크나이저 테스트:")
    
    # 한국어 테스트
    test_texts = [
        "안녕하세요, 세계!",
        "오늘 날씨가 정말 좋네요.",
        "인공지능 기술이 빠르게 발전하고 있습니다.",
        "한국의 전통 문화는 매우 아름답습니다."
    ]
    
    for text in test_texts:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding)
        print(f"\n  원본: {text}")
        print(f"  토큰 ID: {encoding[:10]}{'...' if len(encoding) > 10 else ''}")
        print(f"  토큰 수: {len(encoding)}")
        print(f"  복원: {decoded}")
        
        # 토큰을 하나씩 디코딩해서 보기
        tokens = [tokenizer.decode([token_id]) for token_id in encoding[:5]]
        print(f"  첫 5개 토큰: {tokens}")
    
    # 특수 토큰 정보
    print("\n📌 특수 토큰:")
    print(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"  Vocab size: {len(tokenizer)}")

if __name__ == "__main__":
    try:
        download_kogpt2_tokenizer()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("transformers 설치가 필요합니다: pip install transformers") 