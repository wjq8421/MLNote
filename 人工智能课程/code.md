### Target Encoding

> 对于类别特征col1，特征值为A占m次，其标签为1出现k次，其ctr为k/m。对于稀疏类别，部分类别值出现次数很少，可考虑单独给一个均值的ctr值。

```python
def parameter_ctr(train, test, col, label):
    # train: 数据集
    # col: 类别型特征
    # label：标签值
    new_col = col + '_ctr_' + str(label)
    train[new_col] = 0.0
    
    ctr = train[col+'_le'].value_counts() 
    k_map = {}
    values = train[col+'_le'].unique().tolist()
    values.extend(test[col+'_le'].unique().tolist())
    for elem in set(values):
        try:
            k_map[elem] = train[(train[col+'_le']==elem)&(train['label']==label)].shape[0] / ctr[elem]
        except:
            k_map[elem] = sum(k_map.values()) / len(k_map.values())
    train[new_col] = train[col+'_le'].apply(lambda x: k_map[x])
    test[new_col] = test[col+'_le'].apply(lambda x: k_map[x])
    return train, test

features = ['Parameter6', 'Parameter7', 'Parameter8', 
            'Parameter9', 'Parameter10']
labels = [1, 2, 3, 4]
for feature in features:
    for label in labels:
        print(feature, label)
        trainX, testX = parameter_ctr(trainX, testX, feature, label)
```

