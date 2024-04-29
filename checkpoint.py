import cv2, ffmpeg, ast
import pandas as pd

def checkpoint(frame):
    ffmpeg.input('./data/results/output.mp4').filter('select', f'eq(n,{frame})').output(f'./data/results/cp-{frame}.jpg',vframes=1,loglevel="quiet").run()
#Using saved frame data, find when two hip angle intersect, previous frame is checkpoint

df = pd.read_csv('./data/out.csv')
epochs=[]
stride=[]
prev_rel=None
relation=None
for index, row in df.iterrows():
    #print(row['c1'], row['c2'])
    r_foot=ast.literal_eval(row['r_foot'])
    l_foot=ast.literal_eval(row['l_foot'])
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
    checkpoint(ov_index)

