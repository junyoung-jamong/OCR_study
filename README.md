# OCR_study
OCR관련 논문 내용 번역 및 정리



## Scene Text Detection

[**CRAFT - Character region awareness for text detection**](https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-분석-da987b32609c)

* STD을 위해 각 문자, 그리고 문자 간의 친화도(Affinity)를 예측하여 텍스트 영역을 유연하고 효과적으로 감지할 수 있는 모델을 제안
* 문자 수준(Character-level) Annotation의 부족을 보완하기 위한 프레임워크를 제안



[**PixelLink: Detecting Scene Text via Instance Segmentation**](./STD/PixelLink.md)

* 픽셀 단위의 Dense prediction을 통한 Instance Segmentation 기반의 STD 모델을 제안



## Scene Text Recognition

[**Naver STR Framework - What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis**](./STR/NAVER_STR_Framework.md)

* 기존 STR 모델을 동일한 데이터 집합에서 학습 및 비교하고, 각 모델의 공통점을 기반으로 4단계의 STR 프레임워크를 정의
* STR 스터디의 시작점으로 추천



[**STAR-Net: A SpaTial Attention Residue Network for Scene Text Recognition**](./STR/STAR-Net.md)

* TPS-ResNet-BiLSTMT-CTC 모델