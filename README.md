# 🚗 cars-ai: 중고차 가격 예측 모델

이 프로젝트는 영국 중고차 데이터를 기반으로 한 **차량 가격 예측 AI** 모델입니다.  
PyTorch와 CatBoost 두 가지 모델을 학습하여 성능을 비교한 후, 최종적으로 PyTorch 모델을 채택했습니다.

---

## 📊 데이터셋

- 출처: [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data)
- 주요 컬럼: `engineSize`, `year`, `mileage`, `fuelType`, `brand`, `model`, `price`

---
## 구성 요약

| 파일명 | 설명 |
|--------|------|
| `deep_model_train.py` | PyTorch 학습 코드 |
| `catboost_model_train.py` | CatBoost 학습 코드 |
| `test_predict_pytorch.py` | PyTorch 단일 예측 테스트 |
| `test_predict_catboost.py` | CatBoost 단일 예측 테스트 |
| `evaluate_models.py` | 전체 모델 성능 비교 |

---
## 🛠 사용 패키지

- Python 3.10 이상  
- PyTorch  
- CatBoost  
- scikit-learn  
- pandas, numpy 등
```
pip install -r requirements.txt
```
---

## 🧠 학습

두 가지 모델을 학습합니다:

### PyTorch (딥러닝 기반)
```
python3 deep_model_train.py
```


### CatBoost (트리 기반 부스팅)
```
python3 catboost_model_train.py
```
학습 완료 후 `saved_models/` 디렉토리에 다음 파일들이 저장됩니다:

- `deep_model.pth`: PyTorch 학습 모델  
- `catboost_model.cbm`: CatBoost 학습 모델  
- `scaler.pkl`, `label_encoders.pkl`: PyTorch용 전처리기  
- `scaler_catboost.pkl`, `label_encoders_catboost.pkl`: CatBoost용 전처리기

---

## 🧪 테스트

단일 입력값에 대한 예측 테스트를 실행할 수 있습니다:

### PyTorch 예측 테스트
```
python3 test_predict_pytorch.py
```
![Image](https://github.com/user-attachments/assets/9abd145b-c913-42a8-86bf-79c5f9c80e32)
### CatBoost 예측 테스트
```
python3 test_predict_catboost.py
```
![Image](https://github.com/user-attachments/assets/2b8c68aa-0bfd-4869-800b-e7a2034bb1d7)

---

## 📈 모델 성능 비교

두 모델의 전반적인 예측 성능을 `evaluate_models.py`를 통해 비교할 수 있습니다:

python3 evaluate_models.py

### 평가 결과
![Image](https://github.com/user-attachments/assets/3d6a4d3f-f2d2-4c29-be74-b99f6430cea5)

| 모델     | RMSE    | MAE     | 예측 시간 | 모델 크기     |
|----------|---------|---------|------------|----------------|
| PyTorch  | 2165.72 | 1484.00 | 0.1080초   | 104.60 KB      |
| CatBoost | 2225.53 | 1482.25 | 0.0647초   | 837.84 KB      |

- **RMSE (Root Mean Squared Error)**: 예측값과 실제값의 차이를 제곱한 후 평균을 내고 제곱근을 씌운 값. 값이 작을수록 좋습니다.  
- **MAE (Mean Absolute Error)**: 예측값과 실제값의 절대적인 차이의 평균입니다.  
- **예측 시간**: 전체 데이터셋에 대해 예측을 수행하는 데 걸린 시간입니다.  
- **모델 크기**: 저장된 모델의 용량입니다.

---

## ✅ 최종 모델 선택: PyTorch

CatBoost와 비교해 RMSE가 더 낮고,  
**모델 크기가 작으며 GPU 가속이 가능**하기 때문에  
PyTorch 모델을 **최종 선택**하여 서비스에 활용하기로 결정했습니다.

---

© 2025 cars-ai 프로젝트