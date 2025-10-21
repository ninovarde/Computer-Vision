# Live Face Recognition 

"""
This script implements a basic Live Face Recognition System.
"""

# Import libraries
import threading # Allows to run multiple operations in parallel
import cv2       # The main library used for computer vision tasks

from deepface import DeepFace # Used for face recognition
import matplotlib.pyplot as plt

%matplotlib 

# Intialize the webcam and set the desired resolution
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


face_match = False

file_dir = "path\\for\\reference\\images"
person_name = "name"
reference_img = cv2.imread(file_dir + person_name + ".png")



def generate_reference_image(
        person_name,
        file_dir = file_dir,
        characterize = True
        ): 
    print("\n\nPress 's' to SAVE\n")
    print("Press 'q' to QUIT\n")
    while True:
        ret, frame = cap.read()
        cv2.imshow("video",frame)
        key = cv2.waitKey(1)
        
        if key == ord("s"):
            cv2.imwrite(file_dir + person_name + '.png', frame)
            break
        if key == ord("q"):
            break
        
    cv2.destroyAllWindows()
    
    if characterize:
        try:
            
            objs = DeepFace.analyze(
              img_path = file_dir + person_name + '.png',
              actions = ['age', 'gender', 'race', 'emotion']
            )
            
            info = [
                    objs[0]['age'],objs[0]['dominant_gender'],
                    objs[0]['dominant_race'],objs[0]['dominant_emotion']
                    ]
            print(info)
            face_detected = True
            
        except ValueError:
            print("Subject info cannot be determined.")
            info = 'no info'
            
                 

    # Convert to RGB
    frame_for_plot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Plot with correct colors
    plt.imshow(frame_for_plot)
    plt.axis('off')
    plt.show()
    

    return frame, info
        
    

def check_face(frame,reference_img):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img)['verified']:
            face_match = True 
        else: 
            face_match = False
    except ValueError:
        face_match = False

def live_face_recognition():
    counter = 0
    print("\n\nPress 'q' to QUIT\n\n")
    while True:
        ret, frame = cap.read()
        
        if ret:
            if counter%30 == 0:
                
                try:
                    threading.Thread(
                        target = check_face, 
                        args = (frame.copy(),reference_img.copy())
                        ).start()
                    
                except ValueError:
                    pass
            counter +=1
            
            if face_match:
                cv2.putText(
                    frame, 
                    "MATCH", 
                    (20,450), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, 
                    (0,255,0),
                    3
                    )
            else:
                cv2.putText(
                    frame, 
                    "NO MATCH",
                    (20,450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, 
                    (0, 0,255),
                    3
                    )
                
            cv2.imshow("video",frame)
            
            
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    
    
    cv2.destroyAllWindows()


print("\n\nWelcome!\nLoad reference image: 1\nTake reference image: 2\n\n")
mode = input()

while True:
    
    if mode == '1':
        path = file_dir + person_name + '.png'
        print(f"\nYou decided to load the reference image in \n--> {path}")
        break
    elif mode == '2':
        print("\nYou decided to take a new reference image.")
        person_name = input('\nInsert the person name: ')
        reference_img, reference_info = generate_reference_image(
                person_name=person_name
                )
        break
    else: 
        print('\nNon valid input! Choose between 1 and 2...')
        mode = input()
        
    
print('Activating the Live Face recognition...\n')
live_face_recognition()