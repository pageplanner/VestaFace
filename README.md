


Install Visual Studio Build Tools: Download the Visual Studio Community installer and select "Desktop development with C++."

install cmake
        - cmake.org (this is how windows users should get cmake)
        - apt install cmake (for Ubuntu or Debian based systems)
        - yum install cmake (for Redhat or CenOS based systems)

python -m pip install setuptools
pip install face_recognition opencv-python
pip install git+https://github.com/ageitgey/face_recognition_models
pip install git+https://github.com/ageitgey/face_recognition_models
python -m pip install git+https://github.com/ageitgey/face_recognition_models



Since Python 3.13 is having "bleeding edge" issues with dlib, we will create a 3.11 environment just for this project.

Install Python 3.11 (Side-by-Side): Go to the Python Website and download the "Windows installer (64-bit)". During installation, do NOT check "Add Python to PATH" (this prevents it from interfering with your 3.13 setup).

Create the Virtual Environment: Open your terminal in C:\Users\ben\Desktop\Vestaboard\VestaFace and run:

PowerShell
# Tell the 3.11 launcher to create a folder called 'env'
py -3.11 -m venv venv
Activate it:

PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate
(You will see (venv) appear in front of your command prompt. This means anything you do now stays inside this folder.)

Install the Libraries in the Bubble:

PowerShell
python -m pip install --upgrade pip
python -m pip install face-recognition opencv-python