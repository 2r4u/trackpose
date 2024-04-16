import cv2
import mediapipe as mp
import numpy as np
import ffmpeg, subprocess
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

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

angle_min = 0.0
stride_frame=0
cap = cv2.VideoCapture("./data/images/anthony-tj.mp4")



outfile='./data/results/output.mp4'
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) *0.55;
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.55; 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#TODO make fps adapt to input fps
out = cv2.VideoWriter(outfile,fourcc, 30, (int(w),int(h)))

frames=0
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        frames+=1
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
            
            #check if this frame is max leg extension (stride sync)
            if r_angle_hip+r_angle_knee>angle_min:
                stride_frame=frames
   
            #r_hip_angle = 180-r_angle_hip
            #l_hip_angle = 180-l_angle_hip
            #r_knee_angle = 180-r_angle_knee
            #l_knee_angle =180-l_angle_knee
            
            
            
            # Visualize angle in top left corner
            cv2.putText(image, "rknee: "+str(r_angle_knee), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, "lknee: "+str(l_angle_knee), (50,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)

            cv2.putText(image, "rhip: "+str(r_angle_hip), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, "lhip: "+str(l_angle_hip), (50,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1, cv2.LINE_AA)
            
            #vizualize angle on body
            #cv2.putText(image, str(r_angle_knee),tuple(np.multiply(r_knee, [630,900]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA)
                                

           

            #cv2.putText(image, str(l_angle_knee),tuple(np.multiply(l_knee, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,9,0), 2, cv2.LINE_AA)
                                
            
            #cv2.putText(image, str(r_angle_hip),tuple(np.multiply(r_hip, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                                

            #cv2.putText(image, str(l_angle_hip), tuple(np.multiply(l_hip, [630,900]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                                

            
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1), 
                                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1) 
                                 )               
        
        out.write(rescale_frame(image,100))
        if ret:
            pass
        else:
            break 
    cap.release()
    out.release()
    
#cut video down to sync stride
input_file = ffmpeg.input(outfile)
#TODO: change output name, add some dating/diff system
output_file = ffmpeg.output(input_file.trim(start_frame=stride_frame, end_frame=30*get_length(outfile)), f'{outfile[:len(outfile)-4]}-trimmed.mp4')
ffmpeg.run(output_file)
