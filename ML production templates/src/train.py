from sklearn import preprocessing
import pandas as pd
import os
from sklearn import ensemble

TRAINING_DATA = os.environ.get('TRAINING_DATA')
FOLD = os.environ.get('FOLD')
FOLD_MAPPING = {0:[1,2,3,4],1:[0,2,3,4],2:[0,1,3,4],3:[0,1,2,4],4:[0,1,2,3]}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]
    y_train = train_df.target.values
    y_valid = valid_df.train.values

    train_df = train_df.drop(['id','target','kfold'],axis=1)
    valid_df = valid_df.drop(['id','target','kfold'],axis =1)

label_encoders = []
for c in train_df.columns:
    lbl = preprocessing.LabelEncoder
    lbl.fit(train_df[c].values.to_list() + valid_df[c].values.to_list())
    train_df.loc[:c] = lbl.transform(train_df[c].values.tolist())
    valid_df.loc[:c] = lbl.transform(valid_df[c].values.tolist())
    label_encoders.append((c,lbl))

# data is ready for training

clf = ensemble.RandomForestClassifier(n_jobs=1,verbose=2)
clf.fit(train_df,y_train)
preds = clf.predict_proba(valid_df)[:1]
print(preds)

