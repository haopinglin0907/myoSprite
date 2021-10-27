# myoSprite

Control sprite using myoArmband. The taks of the game is to collect coins using 4 predefined gestures. 

The mappings of the gestures and the direction of the sprite are as follow:

#### Up: Wrist and finger extension

#### Down: Fist

#### Left: Opposition

#### Right: Finger mass extension


## Installation (Anaconda)

Create an environment with packages installation (Use requirements.txt)

After installing the Anaconda, open the Anaconda Prompt and enter the following commands:

conda create -name "env_name" python = 3.8.8

conda activate "env_name"

conda install --file requirements.txt

## Usage

1) Run "python main_calibration.py" to collect within session calibration data (subject-dependent) and train a CNN model that will be used later in main.py. Please follow the instruction to perform the gesture.


2) Run "python main.py", the game will be controlled using the model trained on the calibration data from last step. 
