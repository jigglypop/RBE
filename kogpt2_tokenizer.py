#!/usr/bin/env python3
"""
KoGPT2 토크나이저 인터페이스
"""

import sys
import json
from transformers import PreTrainedTokenizerFast

def main():
    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/skt-kogpt2-base-v2")
    
    if len(sys.argv) < 2:
        print("Usage: python kogpt2_tokenizer.py [encode|decode] <text or token_ids>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "encode":
        # 텍스트를 토큰 ID로 인코딩
        text = " ".join(sys.argv[2:])
        encoding = tokenizer.encode(text)
        print(json.dumps(encoding))
        
    elif mode == "decode":
        # 토큰 ID를 텍스트로 디코딩
        token_ids = json.loads(sys.argv[2])
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(text)
        
    elif mode == "decode_single":
        # 단일 토큰 ID를 텍스트로 디코딩
        token_id = int(sys.argv[2])
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(text)
        
    elif mode == "vocab_size":
        # 어휘 크기 출력
        print(len(tokenizer))
        
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main() 