# EuroSAT Image Classification Report

## **1. Transfer Learning**

본 프로젝트에서는 사전 학습된 **ResNet50**모델을 사용하여 Transfer Learning 기법으로 정확도를 향상시켰습니다. 기존 ResNet50 모델 FC Layer를 추가해 학습하였고, Activation으로 softmax를 사용해 10개의 class로 분류하도록 커스터마이징 한 모델을 사용했습니다.

![model.png](https://github.com/yesjuhee/image-classification/images/model.png)

### **Feature Extraction**

초기 단계에서는 ResNet50 모델의 convolutional base을 고정(frozen)하고, EuroSAT 데이터셋을 사용하여 추가한 상위 계층만 학습시켰습니다.

### **Fine-Tuning**

이후 convolutional base를 해제(unfrozen)하고 더 낮은 학습률로 학습하여 사전 학습된 특징을 새로운 데이터셋에 적응시켰습니다.

## 2. ResNet50

ResNet50(Residual Network with 50 layers)은 이미지 분류 작업에 널리 사용되는 심층 합성곱 신경망입니다. 이 모델은 Residual Connections을 도입하여 역전파 과정에서 기울기 소실 문제를 해결하고, 더 깊은 네트워크에서도 성능 저하를 방지합니다.

ResNet50은 1,000개 클래스에 걸쳐 120만 개 이상의 라벨이 붙은 이미지로 구성된 ImageNet 데이터 세트로 사전 학습되었습니다. 이미지넷은 동물, 사물, 장면 등 광범위한 시각적 개념을 다루고 있어 Transfer Learning을 위한 훌륭한 기반이 됩니다.

## 3. Data Augmentation, Early Stopping, Dropout

정확도 향상을 위해 Data Augmentation, Early Stopping, Dropout 기법을 추가로 사용했습니다.

### 3.1 Data Augmentation

```python
def augment(image, label):
    image, label = preprocess(image, label)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label
```

### 3.2 Early Stopping

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### 3.3 Dropout

```python
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])
```

## 4. Result

### 4.1 Feature Extraction

![feature-extraction.png](https://github.com/yesjuhee/image-classification/images/feature-extraction.png)

Feature Extraction 단계에서는 11번의 epoch가 진행되었습니다. Training accuracy 0.9256, validation accuracy 0.9183에서 중단되었습니다.

### 4.2 Fine-Tuning

![fine-tuning.png](https://github.com/yesjuhee/image-classification/images/fine-tuning.png)

Fine Tuning 단계에서는 12번의 epoch가 진행되었습니다. Training accuracy 0.9791, validation accuracy 0.9609에서 중단되었습니다.

### 4.3 Conclusion

Test 데이터셋으로 최종 평가한 결과입니다.

- Test Accuracy: 0.9696
- Test Loss: 0.0876

본 프로젝트에서는 ResNet50을 활용한 Transfer Learning 및 Fine tuning을 통해 EuroSAT 데이터셋의 위성 이미지를 성공적으로 분류했습니다. Data Augmentation, Drop out, Early Stopping와 같은 추가 기법은 모델의 강건성과 일반화를 더욱 향상시키는 데 기여했습니다. 최종 모델은 테스트 데이터셋에서 96.96%의 높은 정확도를 기록하며, 좋은 일반화 성능을 보여주었습니다.
