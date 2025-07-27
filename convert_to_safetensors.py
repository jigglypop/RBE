#!/usr/bin/env python3
"""
PyTorch 모델을 SafeTensors 형식으로 변환하는 스크립트
실제 모델 가중치를 다운로드하고 변환합니다.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import argparse
from pathlib import Path

def convert_model_to_safetensors(model_id: str, output_dir: str):
    """PyTorch 모델을 SafeTensors로 변환"""
    print(f"🔄 모델 다운로드 중: {model_id}")
    
    # 캐시 디렉토리 설정
    cache_dir = Path(output_dir) / model_id.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 모델과 토크나이저 다운로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print(f"✅ 모델 다운로드 완료")
        print(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # state_dict 추출
        state_dict = model.state_dict()
        
        # SafeTensors로 저장
        output_path = cache_dir / "model.safetensors"
        print(f"💾 SafeTensors로 변환 중...")
        
        # float32로 변환하여 저장
        tensors = {k: v.float().contiguous() for k, v in state_dict.items()}
        save_file(tensors, output_path)
        
        print(f"✅ 변환 완료: {output_path}")
        
        # 토크나이저도 저장
        tokenizer.save_pretrained(cache_dir)
        print(f"✅ 토크나이저 저장 완료")
        
        # 변환된 가중치 정보 출력
        print("\n📋 변환된 레이어 목록:")
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i < 10:  # 처음 10개만 출력
                print(f"  - {name}: {list(tensor.shape)}")
            elif i == 10:
                print(f"  ... 외 {len(state_dict) - 10}개 레이어")
                break
                
        return True
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="PyTorch 모델을 SafeTensors로 변환")
    parser.add_argument("--model", default="EleutherAI/polyglot-ko-1.3b", help="변환할 모델 ID")
    parser.add_argument("--output", default="models/korean_cache", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    # 한국어 모델 목록
    korean_models = [
        "EleutherAI/polyglot-ko-1.3b",
        "skt/kogpt2-base-v2",
        "beomi/KoAlpaca-Polyglot-5.8B",  # 더 큰 모델
    ]
    
    if args.model in korean_models:
        print(f"🇰🇷 한국어 모델 변환 시작: {args.model}")
        success = convert_model_to_safetensors(args.model, args.output)
        
        if success:
            print("\n✅ 모델 변환이 성공적으로 완료되었습니다!")
            print(f"📁 변환된 파일 위치: {args.output}/{args.model.replace('/', '_')}/model.safetensors")
        else:
            print("\n❌ 모델 변환에 실패했습니다.")
    else:
        print(f"⚠️  알 수 없는 모델: {args.model}")
        print(f"지원되는 한국어 모델: {', '.join(korean_models)}")

if __name__ == "__main__":
    main() 