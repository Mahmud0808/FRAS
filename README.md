# Facial Recognition Attendance System GUI

<p align="center">
<a href="https://ibb.co/M994ZTj" target=”_blank”><img src="https://i.ibb.co/TrrSYXn/Screenshot-2022-11-29-223142.png" alt="Screenshot-2022-11-29-223142" border="0" style="width: 48%; margin: 32px;"></a>
<a href="https://ibb.co/GVgk8Ww" target=”_blank”><img src="https://i.ibb.co/zSKbBXc/Screenshot-2022-11-29-223214.png" alt="Screenshot-2022-11-29-223214" border="0" style="width: 48%; margin: 32px;"></a>
</p>

Smart attendance system using facial recognition with GUI.

## Features

#### Find faces in video stream

Find all the faces that appear in each frame.

#### Recognize the faces

Display the name and id of each recognized student in a frame.

#### Marks attendance on Excel Sheet

Marks the attendances of each student appearing in a frame on excel sheet.

#### Admin Panel system

Admin can register, unregister and remove students from database.

#### Search specific attendances

Search specific attendances by date or student id.

## Installation

### Requirements

  * Python 3.3+
  * PyCharm

### Required modules

```
csv
shutil
cv2
os
flask
datetime
numpy
sklearn.neighbors
pandas
joblib
```

## Usage

### Containing Files & Folders

Missing files and folders will be generated automatically.

* `/Attendance` - This folder will contain the generated excel sheets of attendances.
* `/static/face_recognition_model.pkl` - This file will store the trained face recognition model.
* `/static/faces` - This is the folder where we will keep all the pictures of students.
* `/static/resources` - This is the folder from where we will use images and icons for our web gui.
* `/static/haarcascade_frontalface_default.xml` - This is the Haar Cascade face detection algorithm.
* `/templates` - This folder will contain the html files for our web gui.
* `/UserList` - This folder will contain the list of registered and unregistered students.
* `/app.py` - This is our main python program.

#### How to Run

Just run the `app.py` file and voila!

#### Default Username and Password for Admin

Username : `admin`<br>
Password : `12345`

#### Show Debugger Instead of Error Page

just replace `app.run(debug=False)` in app.py with `app.run(debug=True)`
