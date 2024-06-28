"""
2022.9.18 19:55
author:chy
使用LSTM进行预测，共8个特征，步长为60
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Flatten

data_dim = 5
epochs = 30
batch_size = 32

df = pd.read_csv("electric1.csv")
# del df['date']
df_for_training = df[:8401]
df_for_testing = df[8401:]
# df4 = pd.read_csv("power.csv")
# del df4['date']
# df_for_training = df4[:17329]
# df_for_testing = df4[17329:]

'''
可以注意到数据范围非常大，并且它们没有在相同的范围内缩放，
因此为了避免预测错误，让我们先使用MinMaxScaler缩放数据。
(也可以使用StandardScaler)
'''
scaler = MinMaxScaler(feature_range=(0, 1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)


# print(df_for_training_scaled)

def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)


trainX, trainY = createXY(df_for_training_scaled, 60)
testX, testY = createXY(df_for_testing_scaled, 60)

# trainX = np.expand_dims(trainX, axis=0)
# trainY = np.expand_dims(trainY, axis=0)
# testX = np.expand_dims(testX, axis=0)
# testY = np.expand_dims(testY, axis=0)
# print("trainX Shape-- ", trainX.shape)
# print("trainY Shape-- ", trainY.shape)
# print("testX Shape-- ", testX.shape)
# print("testY Shape-- ", testY.shape)
#
# print("trainX[0]-- \n", trainX[0])
# print("trainY[0]-- ", trainY[0])


def build_model():
    model = Sequential()

    # 添加全连接层
    model.add(Dense(200, activation='relu', input_shape=trainX.shape[1:]))
    model.add(Dense(100, activation='relu'))

    # 如果输入数据不是1D的，可以保留Flatten层
    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='Adam', metrics=['mape'])
    # 创建一个 History 对象来捕获训练历史


    # 开始训练模型，并将损失历史保存在 history 变量中
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size)

    # 绘制损失曲线

    return model


    # model = Sequential()
    #
    # model.add(Dense(200, return_sequences=True, input_shape=trainX.shape[1:]))
    # model.add(Dense(100, return_sequences=True))
    #
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))
    #
    # model.compile(loss='mse', metrics=['mape'], optimizer='Adam')
    # return model


model = build_model()
model.fit(trainX,trainY,epochs=epochs, verbose=1, validation_data=(testX,testY),batch_size=batch_size)
prediction = model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-", prediction.shape)
'''
因为在缩放数据时，我们每行有 data_dim 列，现在我们只有 1 列是目标列。
所以我们必须改变形状来使用 inverse_transform
'''
prediction_copies_array = np.repeat(prediction, data_dim, axis=-1)
'''
data_dim列值是相似的，它只是将单个预测列复制了 4 次。所以现在我们有 5 列相同的值 。
'''
print(prediction_copies_array.shape)
"""
这样就可以使用 inverse_transform 函数。
"""
pred = scaler.inverse_transform(np.reshape(prediction_copies_array, (len(prediction), data_dim)))[:, 0]
test=pd.DataFrame(data=pred)
test.to_csv('./datapack/vmd-cnn-am-GRU/BP-electric.csv')
"""
但是逆变换后的第一列是我们需要的，所以我们在最后使用了 → [:,0]。
现在将这个 pred 值与 testY 进行比较，但是 testY 也是按比例缩放的，也需要使用与上述相同的代码进行逆变换。
"""
original_copies_array = np.repeat(testY, data_dim, axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies_array, (len(testY), data_dim)))[:, 0]

print("Pred Values-- ", pred)
print("\nOriginal Values-- ", original)

plt.plot(original, color='red', label='Real Power')
plt.plot(pred, color='blue', label='Predicted Power')
plt.title('Power Prediction')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.show()

# model.save('Model_future_value_LSTM.h5')
# print('Model Saved!')
