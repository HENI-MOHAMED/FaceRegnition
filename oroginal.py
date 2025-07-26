def detect_faces_and_store_embeddings():
    """
    This function detects faces in an image, crops them, and stores the cropped images.
    It also connects to a PostgreSQL database to store the embeddings of the detected faces.
    """
    # importing the cv2 library
    import cv2

    # loading the haar case algorithm file into alg variable
    alg = "haarcascade_frontalface_default.xml"
    # passing the algorithm to OpenCV
    haar_cascade = cv2.CascadeClassifier(alg)
    # loading the image path into file_name variable - replace <INSERT YOUR IMAGE NAME HERE> with the path to your image
    file_name = "test-image.png"
    # reading the image
    img = cv2.imread(file_name, 0)
    # creating a black and white version of the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # detecting the faces
    faces = haar_cascade.detectMultiScale(
        gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(110, 110)
    )

    i = 0
    # for each face detected
    for x, y, w, h in faces:
        # crop the image to select only the face
        cropped_image = img[y : y + h, x : x + w]
        # loading the target image path into target_file_name variable  - replace <INSERT YOUR TARGET IMAGE NAME HERE> with the path to your target image
        target_file_name = 'stored-faces/' + str(i) + '.jpg'
        cv2.imwrite(
            target_file_name,
            cropped_image,
        )
        i = i + 1
        
    # importing the required libraries
    import numpy as np
    from imgbeddings import imgbeddings
    from PIL import Image
    import os
    import json

    # Initialize embeddings dictionary
    embeddings_data = {}
    
    # Path to store embeddings
    embeddings_file = "face_embeddings.json"
    
    # Load existing embeddings if file exists
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)

    for filename in os.listdir("stored-faces"):
        # Skip if embedding already exists
        if filename in embeddings_data:
            continue
            
        # opening the image
        img = Image.open("stored-faces/" + filename)
        # loading the `imgbeddings`
        ibed = imgbeddings(model_path=r'C:\Users\Lenovo\Desktop\work\FaceRegnition\patch32_v1.onnx')
        # calculating the embeddings
        embedding = ibed.to_embeddings(img)
        # Store embedding in dictionary
        embeddings_data[filename] = embedding[0].tolist()
        print(f"Processed {filename}")
    
    # Save embeddings to file
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings_data, f)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the front camera

    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    print("Press SPACE to capture your photo...")
    # loading the `imgbeddings`
    ibed = imgbeddings(model_path=r'C:\Users\Lenovo\Desktop\work\FaceRegnition\patch32_v1.onnx')
    
    # Load stored embeddings
    with open("face_embeddings.json", 'r') as f:
        embeddings_data = json.load(f)
    
    # Get the first stored face embedding
    first_filename = next(iter(embeddings_data))
    r = embeddings_data[first_filename]
    import math

    def cosine_similarity(a, b):
        dot_product = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai ** 2 for ai in a))
        norm_b = math.sqrt(sum(bi ** 2 for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    j=0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
    
            print("Error: Can't receive frame")
            break
            
        # Display the frame
        # cv2.imshow('Camera', frame) 
        
        # Wait for spacebar press - this will capture the image
        
        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # creating a black and white version of the image
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        # detecting the faces
        faces = haar_cascade.detectMultiScale(
            gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(110, 110)
        )
        # get the first face
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # crop the image to select only the face
            cropped_image = cv_img[y:y + h, x:x + w]
            # Convert to PIL Image
            img = Image.fromarray(cropped_image)
        else:
            img = Image.open(file_name)  # fallback to full image if no face detected
        # calculating the embeddings
        embedding = ibed.to_embeddings(img)
        emb = embedding[0].tolist()
        similarity = cosine_similarity(r, emb)
        print("Cosine similarity:", similarity)
        j+=1
        if j > 500 and similarity < 0.94:
            print("No face detected or similarity too low, exiting...")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        if similarity > 0.94:
            print("Face recognized")
            cap.release()
            cv2.destroyAllWindows()
            return True
        

    # Release the camera and destroy windows
def detect_folder():

    import os
    import datetime
    from time import sleep

    folder_path = r"C:\Users\Lenovo\Desktop\New folder"
    s = os.stat(folder_path)
    LAT = datetime.datetime.fromtimestamp(s.st_atime)
    while True:
        try:
            stats = os.stat(folder_path)
            M = datetime.datetime.fromtimestamp(stats.st_atime)
            if M > LAT:
                print("Folder has been modified")
                # Close the folder if it is open using a more reliable PowerShell command
                if detect_faces_and_store_embeddings() == False:
                    os.system('powershell -Command "(New-Object -ComObject Shell.Application).Windows() | Where-Object { $_.Document.Folder.Self.Path -eq \'' + folder_path + '\' } | ForEach-Object { $_.Quit() }"')
                    break
                else:
                    break
                
            
            
        except FileNotFoundError:
            print(f"Error: The folder '{folder_path}' was not found.")

detect_folder()


