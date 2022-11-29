import csv
import shutil
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


# ======= Flask Error Handler =======
@app.errorhandler(404)
@app.errorhandler(401)
@app.errorhandler(500)
def http_error_handler(error):
    return render_template('error.html')


# ======= Flask Assign Admin ========
@app.before_request
def before_request():
    g.user = None

    if 'admin' in session:
        g.user = session['admin']


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
if 'Registered.csv' not in os.listdir('UserList'):
    with open('UserList/Registered.csv', 'w') as f:
        f.write('Name,ID,Section')
if 'Unregistered.csv' not in os.listdir('UserList'):
    with open('UserList/Unregistered.csv', 'w') as f:
        f.write('Name,ID,Section')


# ======= Remove Empty Rows From Excel Sheet =======
def remove_empty_cells():
    dfr = pd.read_csv('UserList/Registered.csv')
    dfu = pd.read_csv('UserList/Unregistered.csv')

    dfr.dropna(inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False)
    dfu.dropna(inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    for file in os.listdir('Attendance'):
        csv = pd.read_csv(f'Attendance/{file}')

        csv.dropna(inplace=True)
        csv.to_csv(f'Attendance/{file}', index=False)


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
    if 'face_recognition_model.pkl' in os.listdir('static'):
        os.remove('static/face_recognition_model.pkl')

    if len(os.listdir('static/faces')) == 0:
        return

    faces = []
    labels = []
    user_list = os.listdir('static/faces')
    for user in user_list:
        for img_name in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{img_name}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# ======= Remove Attendance of Deleted User ======
def remAttendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    for file in os.listdir('Attendance'):
        df = pd.read_csv(f'Attendance/{file}')
        df.reset_index()
        csv_file = csv.reader(open(f'Attendance/' + file, "r"), delimiter=",")

        skip_header = True
        i = 0
        for row in csv_file:
            if not row:
                continue

            if skip_header:
                skip_header = False
                continue

            if str(row[1]) not in list(map(str, dfu['ID'])) and str(row[1]) not in list(map(str, dfr['ID'])):
                df.drop(df.index[i], inplace=True)
                df.to_csv(f'Attendance/{file}', index=False)

            i += 1

    remove_empty_cells()


# ======== Get Info From Attendance File =========
def extract_attendance():
    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')
    df = pd.read_csv(f'Attendance/{datetoday}.csv')

    names = df['Name']
    rolls = df['ID']
    sec = df['Section']
    times = df['Time']
    dates = f'{datetoday}'

    reg = []
    roll = list(rolls)
    for i in range(len(df)):
        if str(roll[i]) in list(map(str, dfu['ID'])):
            reg.append("Unregistered")
        elif str(roll[i]) in list(map(str, dfr['ID'])):
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
    if str(userid) not in list(map(str, df['ID'])):
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
    if f'{datetoday}.csv' not in os.listdir('Attendance'):
        with open(f'Attendance/{datetoday}.csv', 'w') as f:
            f.write('Name,ID,Section,Time')

    remove_empty_cells()
    names, rolls, sec, times, dates, reg, l = extract_attendance()
    return render_template('attendance.html', names=names, rolls=rolls, sec=sec, times=times, l=l,
                           datetoday2=datetoday2)


@app.route('/attendancebtn', methods=['GET'])
def attendancebtn():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('attendance.html', datetoday2=datetoday2,
                               mess='Database is empty! Register yourself first.')

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
            identified_person_name = identified_person.split('$')[0]
            identified_person_id = identified_person.split('$')[1]
            add_attendance(identified_person)
            cv2.putText(frame, f'Name: {identified_person_name}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20),
                        2,
                        cv2.LINE_AA)
            cv2.putText(frame, f'ID: {identified_person_id}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, 'Press Esc to close', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 255), 2,
                        cv2.LINE_AA)
        cv2.imshow('Attendance', frame)
        cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) == 27:
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

    if str(newuserid) not in list(map(str, dfu['ID'])) and str(newuserid) not in list(map(str, dfr['ID'])):
        with open('UserList/Unregistered.csv', 'a') as f:
            f.write(f'\n{newusername},{newuserid},{newusersection}')
    else:
        if str(newuserid) in list(map(str, dfu['ID'])):
            return render_template('adduser.html', mess='You are already in unregistered list.')
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
        cv2.imshow('Collecting Face Data', frame)
        cv2.setWindowProperty('Collecting Face Data', cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')

    if len(os.listdir(userimagefolder)) == 0:
        dfu = pd.read_csv('UserList/Unregistered.csv')
        dfu.drop(dfu.index[-1], inplace=True)
        dfu.to_csv('UserList/Unregistered.csv', index=False)

        remove_empty_cells()

        shutil.rmtree(userimagefolder)
        return render_template('adduser.html', mess='Failed to Capture Photos.')
    else:
        train_model()
        return render_template('adduser.html',
                               mess='Waiting for admin aproval. Currently you are listed as Unregistered.')


# ========== Flask Attendance List ============
@app.route('/attendancelist')
def attendancelist():
    if not g.user:
        return render_template('login.html')

    remove_empty_cells()

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

            if str(row[1]) in list(map(str, dfu['ID'])):
                reg.append("Unregistered")
            elif str(row[1]) in list(map(str, dfr['ID'])):
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

                if str(row[1]) in list(map(str, dfu['ID'])):
                    reg.append("Unregistered")
                elif str(row[1]) in list(map(str, dfr['ID'])):
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

    remove_empty_cells()

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


# ========== Flask Unregister a User ============
@app.route('/unregisteruser', methods=['GET', 'POST'])
def unregisteruser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    row = dfr.iloc[[idx]]
    shutil.move('static/faces/' + dfr.iloc[idx]['Name'] + '$' + dfr.iloc[idx]['ID'] + '$' + dfr.iloc[idx]['Section'],
                'static/faces/' + dfr.iloc[idx]['Name'] + '$' + dfr.iloc[idx]['ID'] + '$None')
    train_model()
    row['Section'] = row['Section'].replace(to_replace='.', value='None', regex=True)

    dfu = dfu.append(row, ignore_index=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    dfr.drop(dfr.index[idx], inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    remove_empty_cells()

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


# ========== Flask Delete a User from Registered List ============
@app.route('/deleteregistereduser', methods=['GET', 'POST'])
def deleteregistereduser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    dfr = pd.read_csv('UserList/Registered.csv')
    username = dfr.iloc[idx]['Name']
    userid = dfr.iloc[idx]['ID']
    usersec = dfr.iloc[idx]['Section']

    if f'{username}${userid}${usersec}' in os.listdir('static/faces'):
        shutil.rmtree(f'static/faces/{username}${userid}${usersec}')
        train_model()

    dfr.drop(dfr.index[idx], inplace=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    remove_empty_cells()

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

    remAttendance()

    if l != 0:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Registered Students: {l}')
    else:
        return render_template('registereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Unregistered User List ============
@app.route('/unregistereduserlist')
def unregistereduserlist():
    if not g.user:
        return render_template('login.html')

    remove_empty_cells()

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
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Unregistered Students: {l}')
    else:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Register a User ============
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])
    sec = request.form['section']

    dfu = pd.read_csv('UserList/Unregistered.csv')
    dfr = pd.read_csv('UserList/Registered.csv')

    row = dfu.iloc[[idx]]

    shutil.move('static/faces/' + dfu.iloc[idx]['Name'] + '$' + dfu.iloc[idx]['ID'] + '$None',
                'static/faces/' + dfu.iloc[idx]['Name'] + '$' + dfu.iloc[idx]['ID'] + '$' + sec)
    train_model()
    row['Section'] = row['Section'].replace(['None'], sec)
    dfr = dfr.append(row, ignore_index=True)
    dfr.to_csv('UserList/Registered.csv', index=False)

    dfu.drop(dfu.index[idx], inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    remove_empty_cells()

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
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Unregistered Students: {l}')
    else:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
                               mess="Database is empty!")


# ========== Flask Delete a User from Unregistered List ============
@app.route('/deleteunregistereduser', methods=['GET', 'POST'])
def deleteunregistereduser():
    if not g.user:
        return render_template('login.html')

    idx = int(request.form['index'])

    dfu = pd.read_csv('UserList/Unregistered.csv')
    username = dfu.iloc[idx]['Name']
    userid = dfu.iloc[idx]['ID']
    usersec = dfu.iloc[idx]['Section']

    print(f'{username}${userid}${usersec}')
    print(os.listdir('static/faces'))
    if f'{username}${userid}${usersec}' in os.listdir('static/faces'):
        shutil.rmtree(f'static/faces/{username}${userid}${usersec}')
        train_model()

    dfu.drop(dfu.index[idx], inplace=True)
    dfu.to_csv('UserList/Unregistered.csv', index=False)

    remove_empty_cells()

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

    remAttendance()

    if l != 0:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=l,
                               mess=f'Number of Unregistered Students: {l}')
    else:
        return render_template('unregistereduserlist.html', names=names, rolls=rolls, sec=sec, l=0,
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
