import cv2, ffmpeg
import pandas as pd

def checkpoint(frame):
    ffmpeg.input('./data/results/output.mp4').filter('select', f'eq(n,{frame})').output(f'./data/results/cp-{frame}.jpg',vframes=1,loglevel="quiet").run()
#Using saved frame data, find when two hip angle intersect, previous frame is checkpoint

df = pd.read_csv('./data/out.csv')
prev_rel=None
relation=None
for index, row in df.iterrows():
    #print(row['c1'], row['c2'])
    relation=(row['l_hip']>row['r_hip'])
    print(relation)
    if index!=1:        
        if prev_rel!=relation:
            checkpoint(index)
            print("check")
    prev_rel=relation
    
