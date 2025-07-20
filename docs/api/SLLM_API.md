# 🚀 **RBE SLLM API 서버 가이드**

## 📖 **개요**

**RBE (Riemannian Basis Encoding) SLLM API 서버**는 푸앵카레 볼 기반의 혁신적인 모델 압축 기술을 활용하여 소형 언어 모델의 효율적인 추론을 제공합니다.

### **핵심 기능**
- 🗜️ **극한 압축**: 100:1 이상의 압축률로 모델 크기 축소
- ⚡ **고속 추론**: 압축된 모델로 실시간 텍스트 생성
- 🎯 **고품질 유지**: RMSE < 0.001로 원본 성능 유지
- 🌐 **REST API**: 웹 서비스 형태의 간편한 접근
- 📊 **성능 벤치마크**: 압축 및 추론 성능 측정

---

## 🔧 **서버 시작**

### **기본 실행**
```bash
cargo run --bin rbe-server
```

### **옵션 설정**
```bash
cargo run --bin rbe-server -- \
  --host 0.0.0.0 \
  --port 8080 \
  --max-tokens 100 \
  --temperature 0.7 \
  --top-p 0.9
```

### **서버 설정 옵션**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | `0.0.0.0` | 서버 호스트 주소 |
| `--port` | `8080` | 서버 포트 번호 |
| `--max-tokens` | `100` | 최대 생성 토큰 수 |
| `--temperature` | `0.7` | 기본 temperature 값 |
| `--top-p` | `0.9` | 기본 top-p 값 |
| `--no-cors` | `false` | CORS 비활성화 |

---

## 🌐 **API 엔드포인트**

### **1. 헬스체크**

서버 상태를 확인합니다.

**요청**
```http
GET /health
```

**응답**
```json
{
  "status": "healthy",
  "service": "RBE API Server"
}
```

**cURL 예시**
```bash
curl http://localhost:8080/health
```

---

### **2. 모델 정보 조회**

현재 로드된 모델의 정보를 조회합니다.

**요청**
```http
GET /model/info
```

**응답** (모델 로드됨)
```json
{
  "name": "kogpt2-base-v2",
  "compression_ratio": 128.5,
  "average_rmse": 0.0008,
  "vocab_size": 50257,
  "hidden_size": 768,
  "num_layers": 12,
  "loaded_at": "2024-01-15 10:30:45 UTC"
}
```

**응답** (모델 없음)
```json
{
  "error": "모델이 로드되지 않았습니다",
  "code": 404
}
```

**cURL 예시**
```bash
curl http://localhost:8080/model/info
```

---

### **3. 모델 로딩**

압축된 RBE 모델을 서버에 로딩합니다.

**요청**
```http
POST /model/load
Content-Type: application/json

{
  "compressed_model_path": "./models/kogpt2_rbe.json",
  "original_model_path": "./models/kogpt2-base-v2"
}
```

**응답** (성공)
```json
{
  "success": true,
  "message": "모델이 성공적으로 로드되었습니다",
  "model_info": {
    "name": "kogpt2-base-v2",
    "compression_ratio": 128.5,
    "average_rmse": 0.0008,
    "vocab_size": 50257,
    "hidden_size": 768,
    "num_layers": 12,
    "loaded_at": "2024-01-15 10:30:45 UTC"
  }
}
```

**cURL 예시**
```bash
curl -X POST http://localhost:8080/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "compressed_model_path": "./models/kogpt2_rbe.json",
    "original_model_path": "./models/kogpt2-base-v2"
  }'
```

---

### **4. 모델 압축**

원본 모델을 RBE 방식으로 압축합니다.

**요청**
```http
POST /model/compress
Content-Type: application/json

{
  "model_path": "./models/kogpt2-base-v2",
  "output_path": "./models/kogpt2_rbe.json",
  "compression_level": 3,
  "block_size": 32
}
```

**요청 파라미터**
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `model_path` | string | ✅ | - | 압축할 원본 모델 경로 |
| `output_path` | string | ✅ | - | 압축된 모델 출력 경로 |
| `compression_level` | number | ❌ | 3 | 압축 레벨 (1-5) |
| `block_size` | number | ❌ | 32 | 블록 크기 |

**응답** (성공)
```json
{
  "success": true,
  "message": "모델이 성공적으로 압축되었습니다",
  "compression_ratio": 128.5,
  "compression_time_seconds": 45.2
}
```

**cURL 예시**
```bash
curl -X POST http://localhost:8080/model/compress \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/kogpt2-base-v2",
    "output_path": "./models/kogpt2_rbe.json",
    "compression_level": 4
  }'
```

---

### **5. 텍스트 생성** ⭐

압축된 모델로 텍스트를 생성합니다.

**요청**
```http
POST /generate
Content-Type: application/json

{
  "prompt": "안녕하세요, 오늘 날씨는",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

**요청 파라미터**
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `prompt` | string | ✅ | - | 생성할 텍스트의 시작 프롬프트 |
| `max_tokens` | number | ❌ | 100 | 최대 생성 토큰 수 |
| `temperature` | number | ❌ | 0.7 | 창의성 조절 (0.0-2.0) |
| `top_p` | number | ❌ | 0.9 | 다양성 조절 (0.0-1.0) |
| `stream` | boolean | ❌ | false | 스트리밍 응답 여부 |

**응답**
```json
{
  "text": "안녕하세요, 오늘 날씨는 맑고 따뜻합니다. 산책하기 좋은 날씨네요.",
  "tokens_generated": 23,
  "generation_time_ms": 450,
  "tokens_per_second": 51.1
}
```

**cURL 예시**
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "안녕하세요, 오늘 날씨는",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

**Python 예시**
```python
import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={
        "prompt": "한국의 전통 음식 중에서",
        "max_tokens": 100,
        "temperature": 0.7
    }
)
print(response.json()["text"])
```

**JavaScript 예시**
```javascript
const response = await fetch('http://localhost:8080/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: '인공지능의 미래는',
    max_tokens: 80,
    temperature: 0.6
  })
});
const result = await response.json();
console.log(result.text);
```

---

### **6. 성능 벤치마크**

압축 및 추론 성능을 측정합니다.

**요청**
```http
POST /benchmark
Content-Type: application/json

{
  "matrix_sizes": [[256, 256], [512, 512], [1024, 1024]],
  "iterations": 10
}
```

**요청 파라미터**
| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `matrix_sizes` | array | ❌ | `[[256,256], [512,512], [1024,1024]]` | 테스트할 행렬 크기들 |
| `iterations` | number | ❌ | 10 | 반복 측정 횟수 |

**응답**
```json
{
  "success": true,
  "message": "벤치마크가 성공적으로 완료되었습니다"
}
```

**cURL 예시**
```bash
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "iterations": 5,
    "matrix_sizes": [[512, 512], [1024, 1024]]
  }'
```

---

## 🚨 **에러 처리**

### **공통 에러 응답 형식**
```json
{
  "error": "에러 메시지",
  "code": 500
}
```

### **주요 에러 코드**
| 코드 | 설명 | 해결 방법 |
|------|------|-----------|
| `404` | 모델이 로드되지 않음 | `/model/load` 엔드포인트로 모델 먼저 로드 |
| `500` | 서버 내부 오류 | 로그 확인 후 서버 재시작 |
| `400` | 잘못된 요청 | 요청 형식 및 파라미터 확인 |

---

## 📊 **성능 특징**

### **압축 성능**
- **압축률**: 50:1 ~ 200:1
- **품질**: RMSE < 0.001 (S급)
- **속도**: 1GB 모델 기준 ~30초

### **추론 성능**
- **지연시간**: 토큰당 ~10ms
- **처리량**: 초당 100 토큰
- **메모리**: 원본 대비 99% 절약

### **지원 모델**
- ✅ GPT-2 계열 (KoGPT-2, GPT-2 base/medium)
- ✅ BERT 계열 (KoBERT, BERT base)
- ✅ T5 계열 (KoT5, T5 small)
- 🔄 LLaMA 계열 (개발 중)

---

## 🛠️ **CLI 도구**

### **모델 다운로드**
```bash
cargo run --bin rbe-cli download skt/kogpt2-base-v2
```

### **모델 압축**
```bash
cargo run --bin rbe-cli compress \
  ./models/kogpt2-base-v2 \
  ./models/kogpt2_rbe.json \
  --level 4 \
  --coefficients 500
```

### **텍스트 생성**
```bash
cargo run --bin rbe-cli generate \
  ./models/kogpt2_rbe.json \
  ./models/kogpt2-base-v2 \
  "안녕하세요" \
  --max-tokens 50
```

### **성능 벤치마크**
```bash
cargo run --bin rbe-cli benchmark --iterations 10 --save results.json
```

### **모델 정보 확인**
```bash
cargo run --bin rbe-cli info ./models/kogpt2_rbe.json
```

---

## 🔗 **통합 예시**

### **전체 워크플로우**
```bash
# 1. 서버 시작
cargo run --bin rbe-server &

# 2. 모델 다운로드
cargo run --bin rbe-cli download skt/kogpt2-base-v2

# 3. 모델 압축
curl -X POST http://localhost:8080/model/compress \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/skt-kogpt2-base-v2",
    "output_path": "./models/kogpt2_rbe.json"
  }'

# 4. 모델 로딩
curl -X POST http://localhost:8080/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "compressed_model_path": "./models/kogpt2_rbe.json",
    "original_model_path": "./models/skt-kogpt2-base-v2"
  }'

# 5. 텍스트 생성
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "한국어 자연어 처리의 미래는",
    "max_tokens": 100
  }'
```

---

## 📚 **추가 리소스**

- 📖 [RBE 원리 설명](../paper/PAPER_POINCARE_RBE.md)
- 🔬 [수학적 배경](../paper/12_11비트_미분_사이클_128비트_푸앵카레볼_수학적_표현.md)
- ⚡ [성능 최적화 가이드](./PERFORMANCE_REPORT.md)
- 🧪 [테스트 케이스](../tests/)

---

## 🤝 **지원 및 문의**

- 📧 이메일: support@rbe-llm.ai
- 📱 GitHub: https://github.com/your-org/rbe-llm
- 💬 Discord: https://discord.gg/rbe-llm

**Happy RBE Coding! 🚀** 