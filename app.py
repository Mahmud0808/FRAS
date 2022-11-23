import csv
import cv2
import os
from flask import Flask, request, render_template, session, redirect, g, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# ======== Flask App ========
app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('error.html')


# ======== Current Date & Time =========
datetoday = date.today().strftime("%d-%m-%Y")
datetoday2 = date.today().strftime("%d %B %Y")

# ======== Capture Video ==========
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# ======= Check and Make Folders ========
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('UserList'):
    os.makedirs('UserList')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/{datetoday}.csv', 'w') as f:
        f.write('Name,ID,Section,Time')
if f'Registered.csv' not in os.listdir('UserList'):
    with open('UserList/Registered.csv', 'w') as f:
        f.write('Name,ID,Section')
if f'Unregistered.csv' not in os.listdir('UserList'):
    with open('UserList/Unregistered.csv', 'w') as f:
        f.write('Name,ID,Section')


# ======= Total Registered Users ========
def totalreg():
    return len(os.listdir('static/faces'))


# ======= Get Face From Image =========
def extract_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray_img, 1.3, 5)
    return face_points


# ======= Identify Face Using ML ========
def identify_face(face_array):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(face_array)


# ======= Train Model Using Available Faces ========
def train_model():
    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    for admin in user_list:
        for img_name in os.listdir(f'static/faces/{admin}'):
            img = cv2.imread(f'static/faces/{admin}/{img_name}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(admin)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# ======== Get Info From Attendance File =========
def extract_attendance():
    df = pd.read_csv(f'Attendance/{datetoday}.csv')
    names = df['Name']
    rolls = df['ID']
    sec = df['Section']
    times = df['Time']
    dates = f'{datetoday}'

    reg = []
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')
    roll = list(rolls)
    for i in range(len(roll)):
        if roll[i] in list(dfu['ID']):
            reg.append("Unregistered")
        elif roll[i] in list(dfr['ID']):
            reg.append("Registered")

    l = len(df)
    return names, rolls, sec, times, dates, reg, l


# ======== Save Attendance =========
def add_attendance(name):
    username = name.split('$')[0]
    userid = name.split('$')[1]
    usersection = name.split('$')[2]
    current_time = datetime.now().strftime("%I:%M %p")

    df = pd.read_csv(f'Attendance/{datetoday}.csv')
    if userid not in list(df['ID']):
        with open(f'Attendance/{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{usersection},{current_time}')


# ======= Flask Home Page =========
@app.route('/')
def home():
    if g.user:
        return render_template('home.html', admin=True, mess='Logged in as Administrator', user=session['admin'])

    return render_template('home.html', admin=False, datetoday2=datetoday2)


# ======== Flask Take Attendance ==========
@app.route('/attendance')
def attendance():
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)


@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('attendance.html', datetoday2=datetoday2,
                               mess='Database is empty, add yourself in user list first.')

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, mess='Camera not available.')

    ret = True
    while ret:
        ret, frame = cap.read()
        if extract_faces(frame) != ():
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)


# ========== Flask Add New User ============
@app.route('/adduser')
def adduser():
    return render_template('adduser.html')


@app.before_request
def before_request():
    g.user = None

    if 'admin' in session:
        g.user = session['admin']


@app.route('/adduserbtn', methods=['GET', 'POST'])
def adduserbtn():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    newusersection = request.form['newusersection']

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        return render_template('adduser.html', mess='Camera not available.')

    userimagefolder = 'static/faces/' + newusername + '$' + str(newuserid) + '$' + str(newusersection)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    if newuserid not in list(dfu['ID']) and newuserid not in list(dfr['ID']):
        with open('UserList/Unregistered.csv', 'a') as f:
            f.write(f'\n{newusername},{newuserid},{newusersection}')
    else:
        if newuserid in list(dfu['ID']):
            return render_template('adduser.html', mess='You are already in pending list.')
        else:
            return render_template('adduser.html', mess='You are already a registered user.')

    i, j = 0, 0
    while 1:
        ignore, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding New User', frame)
        if cv2.waitKey(1):
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('adduser.html', mess='Waiting for admin aproval. Currently you are listed as Unregistered.')


# ========== Flask Attendance List ============
@app.route('/attendancelist')
def attendancelist():
    if not g.user:
        return render_template('login.html')

    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates, reg=reg,
                           l=0)


# ========== Flask Search Attendance by Date ============
@app.route('/attendancelistdate', methods=['GET', 'POST'])
def attendancelistdate():
    if not g.user:
        return render_template('login.html')

    date = request.form['date']

    year = date.split('-')[0]
    month = date.split('-')[1]
    day = date.split('-')[2]

    if f'{day}-{month}-{year}.csv' not in os.listdir('Attendance'):
        names, rolls, sec, times, dates, reg, l = extract_attendance()
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=0,
                               mess="Nothing Found!")
    else:
        names = []
        rolls = []
        sec = []
        times = []
        dates = []
        reg = []
        l = 0

        skip_header = True
        csv_file = csv.reader(open(f'Attendance/{day}-{month}-{year}.csv', "r"), delimiter=",")
        dfu = pd.read_csv('UserList/Unregistered.csv')
        dfr = pd.read_csv('UserList/Registered.csv')

        for row in csv_file:
            if skip_header:
                skip_header = False
                continue

            names.append(row[0])
            rolls.append(row[1])
            sec.append(row[2])
            times.append(row[3])
            dates.append(f'{day}-{month}-{year}')

            if row[1] in list(dfu['ID']):
                reg.append("Unregistered")
            elif row[1] in list(dfr['ID']):
                reg.append("Registered")
            else:
                reg.append("x")

            l += 1

        if l != 0:
            return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg())
        else:
            return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                                   reg=reg, l=l,
                                   totalreg=totalreg(),
                                   mess="Nothing Found!")


# ========== Flask Search Attendance by ID ============
@app.route('/attendancelistid', methods=['GET', 'POST'])
def attendancelistid():
    if not g.user:
        return render_template('login.html')

    id = request.form['id']

    names = []
    rolls = []
    sec = []
    times = []
    dates = []
    reg = []
    l = 0

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    for file in os.listdir('Attendance'):
        csv_file = csv.reader(open('Attendance/' + file, "r"), delimiter=",")

        for row in csv_file:
            if row[1] == id:
                names.append(row[0])
                rolls.append(row[1])
                sec.append(row[2])
                times.append(row[3])
                dates.append(file.replace('.csv', ''))

                if row[1] in list(dfu['ID']):
                    reg.append("Unregistered")
                elif row[1] in list(dfr['ID']):
                    reg.append("Registered")
                else:
                    reg.append("x")

                l += 1

    if l != 0:
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=l,
                               mess=f'Total Attendance: {l}')
    else:
        return render_template('attendancelist.html', names=names, rolls=rolls, sec=sec, times=times, dates=dates,
                               reg=reg, l=l,
                               mess="Nothing Found!")


# ========== Flask Registered User List ============
@app.route('/registereduserlist')
def registereduserlist():
    if not g.user:
        return render_template('login.html')

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


@app.route('/unregisteruser', methods=['GET', 'POST'])
def unregisteruser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    pd.read_csv('UserList/Registered.csv').iloc[[idx]].to_csv('UserList/Unregistered.csv', encoding='utf-8', mode='a',
                                                                index=False, header=False)
    df = pd.read_csv('UserList/Registered.csv')
    df.drop(df.index[idx], inplace=True)
    df.to_csv('UserList/Registered.csv', index=False)

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


@app.route('/deleteregistereduser', methods=['GET', 'POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    df = pd.read_csv('UserList/Registered.csv')
    df.drop(df.index[idx], inplace=True)
    df.to_csv('UserList/Registered.csv', index=False)

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Registered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Pending User List ============
@app.route('/pendinguserlist')
def pendinguserlist():
    if not g.user:
        return render_template('login.html')

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Unregistered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Pending Students: {l}')
    else:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    pd.read_csv('UserList/Unregistered.csv').iloc[[idx]].to_csv('UserList/Registered.csv', encoding='utf-8', mode='a',
                                                                index=False, header=False)
    df = pd.read_csv('UserList/Unregistered.csv')
    df.drop(df.index[idx], inplace=True)
    df.to_csv('UserList/Unregistered.csv', index=False)

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Unregistered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Pending Students: {l}')
    else:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


@app.route('/deletependinguser', methods=['GET', 'POST'])
def deletependinguser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    df = pd.read_csv('UserList/Unregistered.csv')
    df.drop(df.index[idx], inplace=True)
    df.to_csv('UserList/Unregistered.csv', index=False)

    names = []
    rolls = []
    sec = []
    l = 0

    skip_header = True
    csv_file = csv.reader(open('UserList/Unregistered.csv', "r"), delimiter=",")
    for row in csv_file:
        if skip_header:
            skip_header = False
            continue

        names.append(row[0])
        rolls.append(row[1])
        sec.append(row[2])
        l += 1

    if l != 0:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Pending Students: {l}')
    else:
        return render_template('pendinguserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========= Flask Admin Login ============
@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        session.pop('admin', None)
        return redirect(url_for('home', admin=False))

    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '12345':
            session['admin'] = request.form['username']
            return redirect(url_for('home', admin=True, mess='Logged in as Administrator'))
        else:
            return render_template('login.html', mess='Incorrect Username or Password')

    return render_template('login.html')


# ======== Flask Logout =========
@app.route('/logout')
def logout():
    session.pop('admin', None)
    return render_template('login.html')


# ======= Main Function =========
if __name__ == '__main__':
    app.run(debug=True)
