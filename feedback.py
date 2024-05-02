import pandas as pd
from ast import literal_eval

df = pd.read_csv('angles.csv')

def feedback():    
    epochs=[]
    stride=[]
    prev_rel=None
    relation=None
    for index, row in df.iterrows():
        #print(row['c1'], row['c2'])
        r_foot=literal_eval(row['r_foot'])
        l_foot=literal_eval(row['l_foot'])
        relation=(l_foot[0]<r_foot[0])
        stride.append(min(r_foot[1],l_foot[1]))
        #print(r_foot)
        if index!=1:        
            if prev_rel!=relation:
                epochs.append(stride)
                stride=[]
        prev_rel=relation

    flat=[j for sub in epochs for j in sub]
    print(epochs)
    for row in epochs:
        ov_index=flat.index(min(row))
        checkpoints.append


