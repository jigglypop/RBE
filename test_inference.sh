#!/bin/bash

echo "🚀 RBE KoGPT-2 테스트 스크립트"
echo "================================"

# 1. 프로젝트 빌드
echo "📦 프로젝트 빌드 중..."
cargo build --release --bin kogpt2_inference

# 2. 모델 디렉토리 생성
mkdir -p models/skt-kogpt2-base-v2

# 3. 모델 다운로드 (이미 있으면 스킵)
if [ ! -f "models/skt-kogpt2-base-v2/pytorch_model.bin" ]; then
    echo "📥 KoGPT-2 모델 다운로드..."
    ./target/release/kogpt2_inference download
else
    echo "✅ 모델이 이미 존재합니다"
fi

# 4. PyTorch 모델을 safetensors로 변환
if [ ! -f "models/skt-kogpt2-base-v2/model.safetensors" ]; then
    echo "🔄 PyTorch 모델을 safetensors로 변환..."
    python extract_weights.py
else
    echo "✅ Safetensors 파일이 이미 존재합니다"
fi

# 5. RBE 형식으로 변환
echo "🔧 RBE 형식으로 모델 변환..."
./target/release/kogpt2_inference convert

# 6. 간단한 추론 테스트
echo "🤖 추론 테스트..."
./target/release/kogpt2_inference generate -p "안녕하세요, 오늘은" -m 50

# 7. 벤치마크
echo "⚡ 성능 벤치마크..."
./target/release/kogpt2_inference benchmark -i 100

echo "✅ 테스트 완료!" 