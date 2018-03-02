# Tensorflow Datasets start (2)

## Data Import

tensorflow에서 **tf.data API**는 단순할 뿐 아니라 재사용이 가능하고 복잡한 입력 파이프 라인도 구축할 수 있습니다. 에를들어 이미지 모델의 파이프 라인은 분산 파일 시스템의 파일에서 데이터를 들고온 후 각 이미지 데이터셋 섞고 배치를 적용하여 training 시스템이 적용할 수 있습니다.