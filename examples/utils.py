import numpy as np
# lookback表示观察的跨度
def split_data(feature, target, lookback):
    # 将股票数据转换为 numpy 数组
    data_raw = feature
    target_raw = target
    data = []
    target = []

    # 迭代数据，根据 lookback 参数生成输入序列
    # lookback 参数定义了要回溯的时间步长
    for index in range(len(data_raw) - lookback):
        # 从原始数据中截取从当前索引开始的 lookback 长度的数据
        data.append(data_raw[index: index + lookback])
        target.append(target_raw[index: index + lookback])

    # 将列表转换为 numpy 数组
    data = np.array(data)
    target = np.array(target)
    # 计算测试集的大小，这里取数据总量的 20%
    test_set_size = int(np.round(0.2 * data.shape[0]))

    # 计算训练集的大小
    train_set_size = data.shape[0] - test_set_size

    # 分割数据为训练集和测试集
    # x_train 和 x_test 包含除了最后一列外的所有数据
    # y_train 和 y_test 包含最后一列数据
    x_train = data[:train_set_size, :-1, :]
    y_train = target[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = target[train_set_size:, -1, :]

    # 返回分割后的训练数据和测试数据
    return [x_train, y_train, x_test, y_test]


def column_indices(df, colnames):
    """返回列名对应的索引列表"""
    return [df.columns.get_loc(c) for c in colnames if c in df.columns]