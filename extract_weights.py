import torch
import numpy as np
import os
import json

def extract_weights():
    print("PyTorch 모델에서 가중치 추출 중...")
    
    # pytorch_model.bin 로드
    model_path = "models/skt-kogpt2-base-v2/pytorch_model.bin"
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    try:
        # 모델 가중치 로드
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 출력 디렉토리 생성
        output_dir = "models/skt-kogpt2-base-v2/weights"
        os.makedirs(output_dir, exist_ok=True)
        
        # 메타데이터 저장
        metadata = {}
        
        print(f"발견된 레이어: {len(state_dict)} 개")
        
        for name, tensor in state_dict.items():
            # 텐서를 numpy 배열로 변환
            array = tensor.numpy()
            
            # float32로 변환
            if array.dtype != np.float32:
                array = array.astype(np.float32)
            
            # 파일명 생성 (특수문자 제거)
            safe_name = name.replace('.', '_').replace('/', '_')
            
            # numpy 파일로 저장
            np_path = os.path.join(output_dir, f"{safe_name}.npy")
            np.save(np_path, array)
            
            # 메타데이터 저장
            metadata[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "file": f"{safe_name}.npy"
            }
            
            print(f"  ✓ {name}: {array.shape} -> {safe_name}.npy")
        
        # 메타데이터 저장
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ 가중치 추출 완료!")
        print(f"  - 레이어 수: {len(metadata)}")
        print(f"  - 저장 위치: {output_dir}")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    extract_weights() 