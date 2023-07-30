import torch
import os

# 設定資料的路徑
datadir = './data_train/pth'

# 讀取訓練數據
train_data_path = os.path.join(datadir, "train.pth")
train_data = torch.load(train_data_path)
Xtrain = train_data['Xtrain']
Ytrain = train_data['Ytrain']

# 讀取測試數據
test_data_path = os.path.join(datadir, "test.pth")
test_data = torch.load(test_data_path)
Xtest = test_data['Xtest']
Ytest = test_data['Ytest']

print(Xtrain.shape)
print(Ytrain.shape)

print(Xtest.shape)
print(Ytest.shape)


Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).float()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).float()

# 進行維度轉換以符合模型輸入
Xtrain = Xtrain.permute(0, 3, 1, 2)
Xtest = Xtest.permute(0, 3, 1, 2)

# 將Y轉化為類別的整數標籤
_, Ytrain = torch.max(Ytrain, 1)
_, Ytest = torch.max(Ytest, 1)