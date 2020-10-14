from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip
import cv2
from django.contrib import auth
import pyrebase
import face_recognition
import numpy as np


config = {
    "apiKey": "AIzaSyAn0G1qRn7YB42-d8qA1VKXExJ6BZfkj7U",
    "authDomain": "webtest-a3a4b.firebaseapp.com",
    "databaseURL": "https://webtest-a3a4b.firebaseio.com",
    "projectId": "webtest-a3a4b",
    "storageBucket": "webtest-a3a4b.appspot.com",
    "messagingSenderId": "6226427775",
    "appId": "1:6226427775:web:4ec53b061216a2448c644a",
    "measurementId": "G-RK7M8WFQ7V"
}

firebase = pyrebase.initialize_app(config)

fauth = firebase.auth()
database = firebase.database()


face_detector = cv2.CascadeClassifier("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/ass/static/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/ass/static/trainer.yml")

font = cv2.FONT_HERSHEY_SIMPLEX

id = 2 #two persons (e.g. Jacob, Jack)

#idtoken = request.session['uid']
#a = fauth.get_account_info(idtoken)
#a = a['users']
#a = a[0]
#a = a['localId']
#n = database.child('users').child(a).child('details').child('camloginid').get().val()
#y = database.child('users').child(a).child('details').child('campassword').get().val()
#ip = database.child('users').child(a).child('details').child('ipaddress').get().val()
#port = database.child('users').child(a).child('details').child('portno').get().val()
#c = "http://" + n + ":" + str(y) + "@" + ip + ":" + str(port) + "/stream"


names = ['','Thomson']


def get_frame():
    c = "http://ipcam:123456@192.168.0.2:8080/video"
    cam = cv2.VideoCapture(c)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        imgencode = cv2.imencode('.jpg', img)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break


def indexscreen(request):
    try:
        return render(request, 'ass/screen.html')
    except HttpResponseServerError:
        print("error")


@gzip.gzip_page
def dynamic_stream(request, stream_path="video"):
    try:
        return StreamingHttpResponse(get_frame(), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        return "error"

def login(request):
    unsuccessful = 'Please check your credentials'
    successful = 'Sign Up Successful'
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('pass')
        try:
            user = fauth.sign_in_with_email_and_password(email, password)
            return render(request, 'ass/services.html')
        except:
            return render(request, 'ass/login.html',{"us": unsuccessful})
        session_id = user['localId']
        request.session['uid]']=str(session_id)
    return render(request, 'ass/login.html')


def logout(request):
    auth.logout(request)
    return render(request, 'ass/index.html')


def signup(request):
    unsuccessful = 'Please fill in the sign up form'
    successful = 'Sign Up Successful'
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('pass')
        phonenumber = request.POST.get('phonenumber')
        ipcamloginid = request.POST.get('IPLoginId')
        ipcampassword = request.POST.get('IPPassword')
        ipaddress = request.POST.get('IPAddress')
        portno = request.POST.get('IPPort')
        try:
            user = fauth.create_user_with_email_and_password(email, password)

            uid = user['localId']

            data = {"name": name, "phoneno": phonenumber, "camloginid": ipcamloginid, "campassword": ipcampassword,
                    "ipaddress": ipaddress, "portno": portno}

            database.child("users").child(uid).child("details").set(data)

            return render(request, 'ass/login.html', {"s": successful})
        except:
            return render(request, 'ass/signup.html', {"us": unsuccessful})

    return render(request, 'ass/signup.html')


def datafetcher(request):

    userid = request.POST.get('uniqueid')
    username = request.POST.get('unidname')
    data = {"uniqueid":userid,"username":username}
    database.child("uniqueid").set(data)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
    face_detector = cv2.CascadeClassifier('C:/Users/Thomson/Desktop/AutomatedSecuritySystem/ass/static/haarcascade_frontalface_default.xml')
    count = 0
    face_id = userid
    while (True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/Dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('Face Data Fetcher', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break
    cam.release()
    cv2.destroyAllWindows()
    return render(request, 'ass/rekognizer.html')


def trainer(request):
    import os
    from PIL import Image
    import numpy as np

    # Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    path = "C:/Users/Thomson/Desktop/AutomatedSecuritySystem/Dataset"

    def getImagesWithID(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    faces, ids = getImagesWithID(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('C:/Users/Thomson/Desktop/AutomatedSecuritySystem/Trainer/trainer.yml')
    return render(request, 'ass/rekognizer.html')


def rekognizer(request):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:/Users/Thomson/Desktop/AutomatedSecuritySystem/Trainer/trainer.yml')  # load trained model


    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter, the number of persons you want to include
    id = 2  # two persons (e.g. Jacob, Jack)

    names = ['', 'Thomson']  # key in names, start from the second place, leave first empty

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            #cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Rekognizer', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return render(request, 'ass/rekognizer.html')


def security(request):
    import time
    import smtplib

    video_capture = cv2.VideoCapture(0)
    thomson_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1285.jpg")
    thomson_face_encoding = face_recognition.face_encodings(thomson_image)[0]

    wasim_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1297.jpg")
    wasim_face_encoding = face_recognition.face_encodings(wasim_image)[0]

    rizwana_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1292.jpg")
    rizwana_face_encoding = face_recognition.face_encodings(rizwana_image)[0]

    swetha_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1294.jpg")
    swetha_face_encoding = face_recognition.face_encodings(swetha_image)[0]

    known_face_encodings = [
        thomson_face_encoding, wasim_face_encoding, rizwana_face_encoding, swetha_face_encoding
    ]
    known_face_names = [
        "Thomson", "Wasim", "Rizwana", "Swetha"
    ]
    face_locations = []
    face_encodings = []
    face_names = []
    Visitors = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                Visitors.append(name)
        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    Visitors = set(Visitors)
    Visitors = list(Visitors)
    seconds = time.time()
    localtime = time.ctime(seconds)
    f = open("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/sample.txt", "w")
    f.write(localtime)
    for i in Visitors:
        f.write("\n" + i)
    f.close()

    from email.message import EmailMessage

    fromaddress = "chihaya.sasuke@gmail.com"
    msg = EmailMessage()
    msg['Subject'] = 'Visitors List and Time'
    msg['From'] = 'chihaya.sasuke@gmail.com'
    msg['To'] = 'wa1794299@gmail.com'

    with open('C:/Users/Thomson/Desktop/AutomatedSecuritySystem/sample.txt', 'rb') as f:
        file_data = f.read()
        file_name = f.name
    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(fromaddress, '1234567899876543210')
        smtp.send_message(msg)

    Visitors = list(Visitors)
    data = {}
    data["time"] = localtime
    #fire = firebase.FirebaseApplication('https://webtest-a3a4b.firebaseio.com/', None)
    for i in range(len(Visitors)):
        data['Visitor' + str(i)] = Visitors[i]
    database.child("Visitors").set(data)
    return render(request, 'ass/services.html')


def attendance(request):
    import face_recognition
    import cv2
    import numpy as np
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    import datetime

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    # Create a woorksheet
    book = openpyxl.load_workbook("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/AttendanceSheet.xlsx")
    sheet = book["July  2020"]

    # Load a sample pictures and learn how to recognize it.
    thomson_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1285.jpg")
    thomson_face_encoding = face_recognition.face_encodings(thomson_image)[0]

    wasim_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1297.jpg")
    wasim_face_encoding = face_recognition.face_encodings(wasim_image)[0]

    rizwana_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1292.jpg")
    rizwana_face_encoding = face_recognition.face_encodings(rizwana_image)[0]

    swetha_image = face_recognition.load_image_file("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/data/16131a1294.jpg")
    swetha_face_encoding = face_recognition.face_encodings(swetha_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        thomson_face_encoding, wasim_face_encoding, rizwana_face_encoding, swetha_face_encoding
    ]
    known_face_names = {
        3: "Thomson", 4: "Wasim", 5: "Rizwana", 6: "Swetha"
    }

    now = datetime.datetime.now()
    today = now.day
    month = now.strftime("%B")

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    Visitors = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names.get(best_match_index + 3)

                face_names.append(name)
                # seconds= time.time()+19800
                # localtime= str(time.ctime(seconds))
                # Visitors.append(name+'   '+localtime)
                Visitors.append(name)
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Attendance', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    # Details of the Visited persons in the Video
    Visitors = set(Visitors)
    print(Visitors)

    # Attendance sheet
    row_key = 0
    for i in Visitors:
        for j in known_face_names:
            if (known_face_names.get(j) == i):
                row_key = j
                sheet.cell(row=row_key, column=int(today) + 1).value = "Present"
    book.save("C:/Users/Thomson/Desktop/AutomatedSecuritySystem/AttendanceSheet.xlsx")
    return render(request, 'ass/services.html')


def det(request):
    import cv2
    import numpy as np

    net = cv2.dnn.readNet('C:/Users/Thomson/Desktop/ojb dettection/yolov3-tiny.weights',
                          'C:/Users/Thomson/Desktop/ojb dettection/yolov3-tiny.cfg')
    classes = []
    with open('C:/Users/Thomson/Desktop/ojb dettection/coco.names', 'r') as f:
        classes = f.read().splitlines()
    cap = cv2.VideoCapture('C:/Users/Thomson/Desktop/ojb dettection/abc.mp4')
    # img = cv2.imread('img-7.jpg')
    while True:
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(indexes.flatten())

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'ass/services.html')


def hp(request):
    return render(request, 'ass/index.html')


def about(request):
    return render(request, 'ass/about.html')


def features(request):
    return render(request, 'ass/features.html')


def contact(request):
    return render(request, 'ass/contact.html')


def services(request):
    return render(request, 'ass/services.html')