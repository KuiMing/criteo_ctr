# 點擊率預測
## POC
2023/08/11 陳奎銘

---

## 目標與挑戰

- 目標
    - 廣告點擊率是衡量廣告效果的重要指標
    - ***預測點擊率***可以提高廣告投放的效率，節省成本並提高收益
- 挑戰
    - 資料量龐大且特徵多樣
    - 資源與時間有限



---

## 初步 POC 成果

- 從訓練資料集取 2% 作為測試資料
    - Accuracy: 0.785
    - Recall: 0.337
    - Precision: 0.653
    - AUC: 0.788
    - ***Log loss: 0.462***
- Log Loss on Kaggle
    - Private score: 0.468
    - Public score: 0.469
- 有機會降低再降低 Log Loss

---

## 探索資料分析

- 資料欄位
    - 數值型資料：I1~I14
    - 類別型資料：C1~C27
- 資料量
    - `train.txt`： 45,840,617 筆，11.15 GB
    - `text.txt`：6,042,135 筆，1.46 GB


----

## 探索資料分析

- 缺失值比例高
![](media/missing_value.png)

----

## 探索資料分析

- 數值型資料分布較廣
![](media/box_plot.png)

----

## 探索資料分析
- 類別型資料部分欄位的類別眾多
![](media/class_number.png)

---

## 實作過程

- 第一階段
    - 以相對簡單的做法觀察情況
- 第二階段
    - 參考第一階段的觀察
    - 使用深度學習訓練模型

---

## 第一階段嘗試- `LightGBM`

- Light Gradient-Boosting Machine：
    - 基於梯度提升的高效學習框架

![](media/lighgbm.png)

----

## Why `LightGBM`

- 演算法：LightGBM
    - Tree Based Model
    - 可以直接將資料導入做訓練
    - 缺失值有良好的處理方式
    - 訓練速度相對較快
- 特徵工程
    - 只針對類別型資料轉換成 `category`


----

## 模型訓練

- 以小部分樣本針對以下幾項參數做 Grid search，找出較好的參數：
    - learning_rate：控制學習速度
    - feature_fraction：特徵抽樣的比例
    - num_leaves：控制樹的複雜度和大小
- 分段接續訓練模型：
    - 從 `train.txt` 擷取 50% 的資料
    - 每次取 10000 筆資料做訓練
    - 每次紀錄 Feature Importance

----

### 特徵重要性 `Feature Importance`

- 每個欄位的平均重要性，數值越小，代表該欄位資料在此模型的影響力越低：

<img src=media/avg_importance.png width="75%">


----

### 特徵重要性 `Feature Importance`

- 每個欄位重要性為零的次數：

![](media/zero_importance.png)

----

## 結果

- 從訓練資料集取 2% 作為測試資料
    - Accuracy: 0.74
    - Recall: 0.177
    - Precision: 0.479
    - AUC: 0.555
    - Log Los: 0.544
- Log Loss on Kaggle
    - Private score: 0.549
    - Public score: 0.549

---

## 第二階段嘗試- `DeepFM`


- Deep Factorization Machine：
    - 處理大量類別型特徵的問題

![](media/DeepFM.png)

----

## Why `DeepFM`

- DeepFM 能自動學習資料之間交互效應，特別適合處理大量類別型特徵
- 在廣告點擊率預測問題上已證明其有效性



----

## 特徵工程

- 移除部分欄位：
    - 移除重要性較低且缺值情況高達 40 % 以上
    - 39 項 -> 32 項
- 數值型資料處理：
    - 以每個欄位的眾數補缺值
    - 大於 -1 的數值 `+1` 後，取 log
    - 小於 -1 的數值直接設定為 -1
- 類別型資料處理：
    - 以 "-1" 補缺值
    - 針對類別型資料做 Label Encoding

----

## 模型訓練

- 使用 Tensorflow & Keras 進行訓練
- Learning Rate Reducing：
    - Loss 在 10 個 epoch 以內沒有下降，便將 Learning Rate 減半
- Stop Rule：
    - 在 50 個 epoch 以內沒有進步，則停下訓練

----

## 後續調整

- 第一次完成訓練時，發現測試資料的類別型資料有未曾出現過的類別
- 新增一個類別 "-1" 來表示未知類別

----

## 結果

- 從訓練資料集取 2% 作為測試資料
    - Accuracy: 0.785
    - Recall: 0.337
    - Precision: 0.653
    - AUC: 0.788
    - Log loss: 0.462
- Log Loss on Kaggle
    - Private score: 0.468
    - Public score: 0.469


---

## 未來改善方案

- 使用 Batch Normalization
- 增加 Hidden Layer
- 使用更先進的模型
    - DeepLight
        - 在 FM 考慮 Field 之間的交互作用
        - 剪枝：將低於 threshold 的權重設置為零
    - FinalMLP
        - Feature Selection
        - 將 FM 改成 DNN 

---

## 後續預期上線流程

![](media/mlops.png)

---

# Thank you
