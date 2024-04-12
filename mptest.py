import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def rescale_frame(frame, percent=50):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


angle_min = []
angle_min_hip = []

cap = cv2.VideoCapture("./data/images/anthony-tj.mp4")
# Curl counter variables
counter = 0 
min_ang = 0
max_ang = 0
min_ang_hip = 0
max_ang_hip = 0
stage = None




width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./data/results/output_video_.mp4', fourcc, 24, size)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame_ = rescale_frame(frame, percent=55)
        
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
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # right leg
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            
            # Get coordinates
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
           
            
            # Calculate angles
            r_angle_hip = calculate_angle(r_shoulder, r_hip, r_knee)
            l_angle_hip = calculate_angle(l_shoulder, l_hip, l_knee)

            r_angle_knee = calculate_angle(r_hip, r_knee, r_ankle) #Knee joint angle
            r_angle_knee = round(r_angle_knee,2)

            l_angle_knee = calculate_angle(l_hip, l_knee, l_ankle) #Knee joint angle
            l_angle_knee = round(l_angle_knee,2)
   
            #r_hip_angle = 180-r_angle_hip
            #l_hip_angle = 180-l_angle_hip
            #r_knee_angle = 180-r_angle_knee
            #l_knee_angle =180-l_angle_knee
            
            
            
            # Visualize angle
            """cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )"""

            cv2.putText(image, str(r_angle_knee), 
                           tuple(np.multiply(r_knee, [630,900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(l_angle_knee), 
                           tuple(np.multiply(l_knee, [630,900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, str(r_angle_hip), 
                           tuple(np.multiply(r_hip, [630,900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA
                                )

            cv2.putText(image, str(l_angle_hip), 
                           tuple(np.multiply(l_hip, [630,900]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA
                                )

            
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1) 
                                 )               
        
        out.write(image)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            #break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
