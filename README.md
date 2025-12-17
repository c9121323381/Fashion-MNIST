# Fashion‑MNIST 深度學習實作報告（TensorFlow / Keras）

本報告為 **Fashion‑MNIST 神經網路實作的完整逐步學習**，適合剛接觸深度學習、已具備基本 Python 能力的學習者。

內容依照實際操作流程編排，可直接在 **Google Colab** 或 **Jupyter Notebook** 中執行。

參考資料:https://simplelearn.tw/deep_learning_neural-networks-fashion-mnist/

--


## 一、目標說明

本次報告學習的目標是：

> **建立一個深度學習模型，能夠自動辨識圖片中的服飾種類**。

我們使用的資料集是 **Fashion‑MNIST**，其特性如下：

* 每張圖片大小為 `28 × 28`
* 灰階影像（單一通道）
* 總共有 `10` 個分類
* 訓練資料：`60,000` 張
* 測試資料：`10,000` 張

---

## 二、開發環境準備

### 1. 建議環境

* Google Colab（最推薦，無需安裝）
* 或 Jupyter Notebook（Python 3.8 以上）

### 2. 安裝必要套件

若使用本機環境，請先安裝：

```bash
pip install tensorflow matplotlib
```
### 3. 在 Google Colab 將執行環境改成 GPU

開啟執行階段設定

在 Colab 上方選單點：

執行階段 → 變更執行階段類型

<img width="898" height="811" alt="A1" src="https://github.com/user-attachments/assets/074feac7-9da2-4dd4-84a2-7a46810839b0" />


---

## 三、載入套件與資料集

### 1. 匯入 Python 套件

```python
import tensorflow as tf
import matplotlib.pyplot as plt
```
<img width="475" height="161" alt="A2" src="https://github.com/user-attachments/assets/e6799f52-8b67-4145-ad84-6355612bdd6d" />


### 2. 載入 Fashion‑MNIST

```python
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```
<img width="1335" height="339" alt="A3" src="https://github.com/user-attachments/assets/d39f5433-01c7-49e6-9309-bd4aa2773988" />


此時資料已自動下載並分為：

* `training_images`, `training_labels`
* `test_images`, `test_labels`

---

## 四、資料探索與理解


### 1. 顯示一張訓練圖片

```python
plt.imshow(training_images[0])
print(training_labels[0]) #列印出第一個訓練標籤
print(training_images[0]) #列印出第一個訓練圖像的像素數據
```

目的：

* 確認圖片內容
* 理解標籤代表的類別(9對應類別為Ankle boot)

### 分類標籤對應

| Label | 類別            |
| ----- | ------------- |
| 0     | T‑shirt / top |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |
---
<img width="636" height="621" alt="A4" src="https://github.com/user-attachments/assets/d17fd7a0-1269-4f9c-a798-f5f98aa2ea58" />

<img width="319" height="97" alt="A5" src="https://github.com/user-attachments/assets/f6050d02-0f0b-487b-9776-f3af9c25859c" />

<img width="982" height="648" alt="A6" src="https://github.com/user-attachments/assets/9a60e46f-50fc-4e69-ba57-7cf334baa541" />


## 五、資料前處理（Normalization）

### 為什麼要正規化？

* 原始像素值範圍：`0 ~ 255`
* 神經網路在小數範圍（`0 ~ 1`）訓練較穩定

### 執行正規化

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```
 <img width="648" height="103" alt="A7" src="https://github.com/user-attachments/assets/7ff9f885-df33-42c1-9664-a05675243d08" />

---

## 六、建立神經網路模型

### 模型架構說明

本文使用 **全連接神經網路（MLP）**，結構如下：

1. Flatten：將 2D 影像轉為 1D → 把 28×28 的影像展平成 784 個特徵
2. Dense(128)：隱藏層，使用 ReLU → 128 個神經元
3. Dense(10)：輸出層，使用 Softmax → 對應 10 個類別，使用 Softmax 使輸出成機率分布* De

### 建立模型

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
<img width="1688" height="303" alt="A8" src="https://github.com/user-attachments/assets/cc20b910-6590-404d-936f-511f6661167a" />


## 七、編譯模型

### 編譯參數說明

* Optimizer：Adam（自動調整學習率）
* Loss Function：sparse_categorical_crossentropy（分類問題常用）
* Metric：accuracy（準確度）

### 編譯程式碼

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

<img width="639" height="199" alt="A9" src="https://github.com/user-attachments/assets/85ef838f-547a-417c-84ff-b547236e3bed" />

---

## 八、訓練模型

訓練時模型會學習如何把圖像對應到正確的標籤。

### 開始訓練

```python
model.fit(training_images, training_labels, epochs=5)
```
<img width="1223" height="413" alt="A10" src="https://github.com/user-attachments/assets/bfadbd5e-fd5d-49db-a1be-21ccaeafca4f" />

### 觀察重點

* loss 是否逐漸下降
* accuracy 是否逐漸上升

---

## 九、模型評估

### 使用測試資料評估

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
```
<img width="1174" height="214" alt="A11" src="https://github.com/user-attachments/assets/aa3ec62d-7f36-4d89-a353-0ec41dfd3eb9" />

此結果代表模型在「未看過的資料」上的表現。

---

## 十、模型儲存與載入

### 儲存模型

```python
model.save('fashion_mnist_model.keras')
```

### 載入模型

```python
loaded_model = tf.keras.models.load_model('fashion_mnist_model.keras')
```
<img width="1677" height="184" alt="A12" src="https://github.com/user-attachments/assets/369f1c00-fdd5-4373-82c6-b6a0f04ed8c2" />

**.h5更改成.keras**

.h5 是舊版格式，雖可用但將被淘汰；官方建議改用 .keras，可保存完整資訊並與 TensorFlow 更相容。

## 十一、查看模型結構

```python
model.summary()
```
<img width="1058" height="468" alt="A13" src="https://github.com/user-attachments/assets/e03e7f3e-3c21-4b84-8f41-2129b18ac8b1" />


可查看：

* 每一層的輸出形狀
* 參數數量

---

## 十二、延伸學習方向

1. 嘗試增加 Dense 層或神經元數量
2. 改用 **卷積神經網路（CNN）**
3. 使用 Dropout 防止過擬合
4. 視覺化預測結果（prediction vs ground truth）

---

## 十三、總結

透過本次報告學習，我完成：

* 完整的資料載入與前處理
* 建立與訓練神經網路模型
* 評估與保存模型

這是一個標準的 **深度學習入門實戰流程**，未來可以直接套用到其他影像分類問題。
