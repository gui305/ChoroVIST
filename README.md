# **ChoroVIST**
### **ChoroVIST** is an interface to analyze choroid metrics such as Vascular Index and Thickness and performs choroid region segmentations using a backend code called **Choroidalyzer**, which is also available on GitHub on https://github.com/justinengelmann/Choroidalyzer . The interface is implemented with a simple database and includes several useful features that make it easy to use and highly relevant for medical professionals.

# App Interface Overview

## ChoroVIST Welcome Screen
### This is the welcome screen of the app, displaying the main icons that reference the app's icon and the educational institutions responsible for supporting its development.
![Welcome Screen](https://github.com/user-attachments/assets/ce603e6a-8b02-4c33-8899-e9592a425a91)


## ChoroVIST Main Screen
### This is the main screen for daily use by medical professionals, allowing the analysis of new exams as well as searching through the patient's history for previous uploaded exams.
![Main Screen](https://github.com/user-attachments/assets/25bd8764-46f2-4f97-84fd-7f03c11c69cf)


## Chorovist DataBase
### This is the database with which the app communicates, implemented in Excel to better suit the medical environment and provide easier navigation and interpretation.
![DataBase](https://github.com/user-attachments/assets/6cb23d30-feb2-45a7-95ff-207d56da65d3)

## Generated Clinical Report
### This is an example of a generated medical report from the app, ready for printing and clinically relevant.
![DataBase](https://github.com/user-attachments/assets/efecece9-fe9c-451b-9cd4-1ffaa726e2ae)

# 🔧 Installation

## 🛠️ Step 1: Create and Activate a Conda Environment

First, you'll need to create a dedicated conda environment for ChoroVIST. 
If you haven't installed it on your computer yet, we recommend the lightweight [Miniconda](https://www.anaconda.com/download/success).

During installation, make sure to:

**.** Tick the box that says "Add Miniconda to my PATH environment variable" (if available).

**.** Choose the option to register Miniconda as your default Python.

The creation of this environment isolates the dependencies for ChoroVIST and ensures they won't conflict with other projects.

### 1.1 Open your terminal or the Miniconda/Conda PowerShell (depending on what you installed)

- **If you're on Windows**:
  - Click the **Start menu** (or press the Windows key).
  - Type and open `terminal`
  - Open your Conda Prompt

- **If you're on macOS or Linux**:
  - Open the **Terminal** application (you can find it via Spotlight or your applications menu).
  - Then type conda commands directly as shown in the next steps.

### 1.2 Create the Conda Environment

Run the following command to create a new environment named `chorovist` with Python 3.10:

```bash
conda create -n chorovist python=3.10
```
### 1.3 Activate the Environment

Once the environment is created, activate it by running:

```bash
conda activate chorovist
```
## ⚡ Step 2: Install Required Dependencies

After activating the environment, you'll need to install the necessary dependencies to run ChoroVIST.

```bash
pip install torch torchvision torchaudio matplotlib tqdm pandas scikit-image scipy openpyxl oct_converter tkcalendar
```

## ⚙️ Step 3: Set Up the Application

Once the environment is ready and dependencies are installed, ChoroVIST should be ready to run.

### 3.1 Navigate to Your Desired Folder

Before cloning the repository, use the terminal to move to the folder where you want ChoroVIST to be saved.

- For example, if you want to place it in your **Documents** folder:

```bash
cd C:\Users\YourName\Documents
```

### 3.2 Clone the Repository

> ⚠️ Make sure Git is installed on your system.  
> You can check by running:
>
> ```bash
> git --version
> ```
>
> If it's not installed, download it from [git-scm.com](https://git-scm.com/downloads).  

If you haven't cloned the repository yet, you can do so by running the following command in your desired directory:

```bash
git clone https://github.com/gui305/ChoroVIST.git
```
### 3.3 Enter the ChoroVIST Directory

After cloning the repository, navigate into the `ChoroVIST` folder using the terminal:

```bash
cd ChoroVIST
```

### 3.4 Run ChoroVIST
Now that you are in the repository directory, you can run the application using Python:

```bash
python ChoroVIST.py
```

## 🖥️ Optional Step 4: Compile the Application into an Executable App

### 4.1 Install PyInstaller
First, install PyInstaller using pip:

```bash
pip install pyinstaller
```

### 4.2 Compile the Application
Run the following PyInstaller command to compile the application into an executable. This will bundle all dependencies, include necessary assets, and create a single executable file. This could take a few minutes.

```bash
pyinstaller --onefile --windowed --hidden-import=skimage.measure --icon=coroide.ico --add-data "FMUL.png;." --add-data "Tecnico.png;." --add-data "OLHO.png;." --add-data "coroide.ico;." ChoroVIST.py
```

### 4.3 Run the Executable
After compiling, you'll find the ChoroVIST executable inside the dist folder. Simply double-click to run the application like a normal app. You can change the app location understanding that the app will run over the ChoroVIST Database and Datafiles that you have in the directory of the exec.


## 🩺 Usage and Clinical Application

### Everyone is free to use the app for clinical purposes and to edit the code for further investigation or personal changes. 
