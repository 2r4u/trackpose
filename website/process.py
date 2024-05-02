import cv2 
import mediapipe as mp
import numpy as np
from ast import literal_eval
import pandas as pd
import matplotlib
matplotlib.use('agg')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def calculate_angle(a, b, c):
  a = np.array(a)  # First
  b = np.array(b)  # Mid
  c = np.array(c)  # End

  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0])
  angle = np.abs(radians * 180.0 / np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle


def rescale_frame(frame, percent):
  width = int(frame.shape[1] * percent / 100)
  height = int(frame.shape[0] * percent / 100)
  dim = (width, height)
  return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)



def process(source):
    cap = cv2.VideoCapture(source)
  
    OUTPUT_FILE = './results/output.webm'
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.55
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.55
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (int(w), int(h)))
    usr_angles = {
          'r_elbow': [],
          'l_elbow': [],
          'r_knee': [],
          'l_knee': [],
          'r_foot': [],
          'l_foot': [],
          'r_heel':[],
          'l_heel':[]
    }
      
      
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():  
          ret, frame = cap.read()
          if frame is not None:
            frame_ = rescale_frame(frame, 55)
          # Recolor image to RGB
          image = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False
      
          # Make detection
          results = pose.process(image)
      
          # Recolor back to BGR
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
          # Extract landmarks
          try:
            landmarks = results.pose_landmarks.landmark
      
            # format: (bodypart) = [landmarks[mp_pose.PoseLandmark.(BODY_PART).value].x,landmarks[mp_pose.PoseLandmark.(BODY_PART).value].y]
      
            #shoulders
            r_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            l_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            ]
            # right leg
            r_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            ]
            r_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            ]
            r_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
            ]
            r_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]
            l_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
            ]

            # Get coordinates
            l_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            l_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            ]
            l_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]
            #TODO add wrist
            r_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            ]
            l_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            ]
            # Calculate angles
            r_angle_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)
            l_angle_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
      
            r_angle_knee = calculate_angle(r_hip, r_knee, r_ankle)  #Knee joint angle
            r_angle_knee = round(r_angle_knee, 2)
      
            l_angle_knee = calculate_angle(l_hip, l_knee, l_ankle)  #Knee joint angle
            l_angle_knee = round(l_angle_knee, 2)
      
            #check if this frame is max leg extension (stride sync)
            #leg_frames.append(max([r_angle_knee+r_angle_hip,l_angle_hip+l_angle_knee]))
            #add angles to dictionary
            usr_angles['r_elbow'].append(r_angle_elbow)
            usr_angles['l_elbow'].append(l_angle_elbow)
            usr_angles['r_knee'].append(r_angle_knee)
            usr_angles['l_knee'].append(l_angle_knee)
            usr_angles['r_foot'].append(
                [
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
            ]
            )
            usr_angles['l_foot'].append(
                [
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
            ]
            )
            usr_angles['r_heel'].append(
                [
                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y
                ]
            )
            usr_angles['l_heel'].append(
                [
                landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y
                ]
            )
            #r_hip_angle = 180-r_angle_hip
            #l_hip_angle = 180-l_angle_hip
            #r_knee_angle = 180-r_angle_knee
            #l_knee_angle =180-l_angle_knee
      
            # Visualize angle in top left corner
            cv2.putText(image, "rknee: " + str(r_angle_knee), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "lknee: " + str(l_angle_knee), (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
      
            cv2.putText(image, "rhip: " + str(r_angle_elbow), (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, "lhip: " + str(l_angle_elbow), (50, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
      
            #TODO integrate code with flask page, create function to send feedback in feed on page
      
            #vizualize angle on body
            #cv2.putText(image, str(r_angle_knee),tuple(np.multiply(r_knee, [630,900]).astype(int)),cv2.FON:T_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA)
      
            #cv2.putText(image, str(l_angle_knee),tuple(np.multiply(l_knee, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA)
      
            #cv2.putText(image, str(r_angle_hip),tuple(np.multiply(r_hip, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
      
            #cv2.putText(image, str(l_angle_hip), tuple(np.multiply(l_hip, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
      
            # Render detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255),
                                       thickness=2,
                                       circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 0),
                                       thickness=2,
                                       circle_radius=1))
      
            #if abs(r_angle_hip - l_angle_hip)<=2:
            #   count+=1
            #  cv2.imwrite(f'./data/results/cp-{count}.jpg',rescale_frame(image,100))
            # crop video to fit csv
            #if start==False:
            #start=True
            #start_frame=count
            out.write(rescale_frame(image, 100))
      
          except:
            pass
            #count+=1
      
          if ret:
            pass
          else:
            break
        cap.release()
        out.release()
    #save in dataframe, crop to video, save as csv (for comparison)

    df=pd.DataFrame(usr_angles)
    critique={
            'r_elbow':False,
            'l_elbow':False,
            'heel_strikes':0,
            'r_knee':True,
            'l_knee':True
            }
    heel_strikes=0
    el_angle_l=df.loc[:, 'l_elbow'].mean()
    el_angle_r=df.loc[:, 'r_elbow'].mean()
    if 75<=abs(el_angle_l)<=105:
        critique['l_elbow']=True
    if 75<=abs(el_angle_r)<=105:
        critique['r_elbow']=True
    rels=[]
    epochs=[]
    stride=[]
    prev_rel=None
    relation=None
    for index, row in df.iterrows():
        #print(row['c1'], row['c2'])        l_foot=ast.literal_eval(row['l_foot'])
        relation=(row['l_foot'][0]<row['r_foot'][0])
        rels.append(relation)
        stride.append(min(row['l_foot'][1],row['r_foot'][1]))
        #print(r_foot)
        if index!=1:        
            if prev_rel!=relation:
                epochs.append(stride)
                stride=[]
        prev_rel=relation

    flat=[j for sub in epochs for j in sub]
    for row in epochs:
        ov_index=flat.index(min(row))
        if rels[ov_index]:
            if 180-df.loc[ov_index,'r_knee']>40:
                critique['r_knee']=False 
            if df.loc[ov_index,'r_heel'][1]<min(row):
                heel_strikes+=1
        else:
            if 180-df.loc[ov_index,'l_knee']>40:
                critique['l_knee']=False
            if df.loc[ov_index,'l_heel'][1]<min(row):
                heel_strikes+=1
    critique['heel_strikes']=heel_strikes
    return critique
    #fig=df.plot(use_index=True)
    #fig.figure.savefig('./static/images/fig.png')
    #idx_col=[i for i in range(len(df))]
    #df.insert(0, 'index', pd.Series(idx_col))
    #op=df.to_json(orient='values')
    #return op
    
    
