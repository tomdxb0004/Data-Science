import pandas as pd
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    df = pd.read_csv("C:\\Users\\Tom Joseph\\Documents\\ML projects _AK\\Data-Science\\ML production templates\\input\\train.csv")
    df['KFold'] = -1
    df.sample(frac = 1).reset_index(drop = True)

    kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    for fold,(train_idx,val_idx) in enumerate(kf.split(X=df,y=df.target.values)):
        print(len(train_idx),len(val_idx))
        df.loc[val_idx,'KFold'] = fold
    df.to_csv("C:\\Users\\Tom Joseph\\Documents\\ML projects _AK\\Data-Science\\ML production templates\\input\\train_Kfolds.csv",index = False)




