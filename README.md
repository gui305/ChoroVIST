# **ChoroVIST**
### **ChoroVIST** is an interface to analyze choroid metrics such as Vascular Index and Thickness and performs choroid region segmentations using a backend code called **Choroidalyzer**, which is also available on GitHub on https://github.com/justinengelmann/Choroidalyzer . The interface is implemented with a simple database and includes several useful features that make it easy to use and highly relevant for medical professionals.o use

## ChoroVIST Welcome Screen
![Welcome Screen](https://github.com/user-attachments/assets/ce603e6a-8b02-4c33-8899-e9592a425a91)

## ChoroVIST Main Screen
![Main Screen](https://github.com/user-attachments/assets/a5a14f94-06ca-487e-85ab-1549798fd427)

## Chorovist DataBase and Generated Report
![Main Screen](https://github.com/user-attachments/assets/ce603e6a-8b02-4c33-8899-e9592a425a91)

## 🔧 Installation

### 🛠️ Step 1: Create and Activate a Conda Environment

First, you'll need to create a dedicated conda environment for ChoroVIST. This isolates the dependencies for ChoroVIST and ensures they won't conflict with other projects.

#### 1.1 Create the Conda Environment

Run the following command to create a new environment named `chorovist` with Python 3.10:

```bash
conda create -n chorovist python=3.10
```
#### 1.2 Activate the Environment

Once the environment is created, activate it by running:

```bash
conda activate chorovist
```
### ⚡ Step 2: Install Required Dependencies

After activating the environment, you'll need to install the necessary dependencies to run ChoroVIST.

```bash
pip install torch torchvision torchaudio matplotlib tqdm pandas scikit-image scipy openpyxl oct_converter tkcalendar
```

### ⚙️ Step 3: Set Up the Application

Once the environment is ready and dependencies are installed, ChoroVIST should be ready to run.

#### 3.1 Clone the Repository (If You Haven't Already)

If you haven't cloned the repository yet, you can do so by running the following command in your desired directory:

```bash
git clone https://github.com/gui305/ChoroVIST.git
```

#### 3.2 Run ChoroVIST
Now that you are in the repository directory, you can run the application using Python:

```bash
python ChoroVIST.py
```

### 🖥️ Optional Step 4: Compile the Application into an Executable App

#### 4.1 Install PyInstaller
First, install PyInstaller using pip:

```bash
pip install pyinstaller
```

#### 4.2 Compile the Application
Run the following PyInstaller command to compile the application into an executable. This will bundle all dependencies, include necessary assets, and create a single executable file. This could take a few minutes.

```bash
pyinstaller --onefile --windowed --hidden-import=skimage.measure --icon=coroide.ico --add-data "FMUL.png;." --add-data "Tecnico.png;." --add-data "OLHO.png;." --add-data "coroide.ico;." ChoroVIST.py
```

#### 4.3 Run the Executable
After compiling, you'll find the ChoroVIST executable inside the dist folder. Simply double-click to run the application like a normal app. You can change the app location understanding that the app will run over the ChoroVIST Database and Datafiles that you have in the directory of the exec.
