# ML ZoomCamp Capstone Project

Hello guys, this repository contains the work carried out as a part of the Capstone Project for the course Machine Learning ZoomCamp. This course is organized and conducted by Mr. Alexey Grigorev.

#### Links to this course material:
[DataTalksClub](https://datatalks.club/courses/2021-winter-ml-zoomcamp.html) <br>
[Machine Learning ZoomCamp Course Material](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)

## Contents
 1. Problem Description
 2. Files Description
 3. Virtual Environment and Packages
 4. Deploy model as Web Service
 5. Deploy model on local machine



## 1. Problem Description and Background
In today's world, Cardiovascular diseases (CVDs) are one of the most common causes of death.It is a class of diseases that involve the heart or blood vessels.Together CVD resulted in 17.9 million deaths (32.1%) in 2015, up from 12.3 million (25.8%) in 1990. Deaths, at a given age, from CVD are more common and have been increasing in much of the developing world, while rates have declined in most of the developed world since the 1970s. Coronary artery disease and stroke account for 80% of CVD deaths in males and 75% of CVD deaths in females. Most cardiovascular disease affects older adults. In the United States 11% of people between 20 and 40 have CVD, while 37% between 40 and 60, 71% of people between 60 and 80, and 85% of people over 80 have CVD. The average age of death from coronary artery disease in the developed world is around 80 while it is around 68 in the developing world. Diagnosis of disease typically occurs seven to ten years earlier in men as compared to women.
Amongst all types of CVDs, heart failure is one of the most common effect observed. 
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

- The [data](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) that I have used contains 12 features that can be used to predict mortality by heart failure.
- This project is a binary classification problem.
- We will predict the deaths due to heart failures using different models (our target variable being Death_Event).
- Target variable is the variable that is or should be the output. In our study our target variable is Death Event in the contex of determining whether anybody is likely to die due to heart failures based on the input parameters like gender, age and various test results or not.
- Lastly, we will build a variety of Classification models and compare the models giving the best prediction on Heart Failures.


### Features of this dataset
- age: Person's age.
- anaemia: Decrease of red blood cells or hemoglobin (boolean).
- creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L).
- diabetes: If the patient has diabetes (boolean).
- ejection_fraction: Percentage of blood leaving the heart at each contraction (percentage).
- high_blood_pressure: If the patient has hypertension (boolean).
- platelets: Platelets in the blood (kiloplatelets/mL).
- serum_creatinine: Level of serum creatinine in the blood (mg/dL).
- serum_sodium: Level of serum sodium in the blood (mEq/L).
- sex: Woman or man (binary).
- smoking: If the patient smokes or not (boolean).
- time: Follow-up period (days).
- DEATH_EVENT: If the patient deceased during the follow-up period (boolean).


## 2. Files Description
- ### Capstone_Project_with_output.ipynb
  This file contains the following sections with their respective outputs: <br>
      - Libraries <br>
      - Load Data <br>
      - Data Pre-processing <br>
      - EDA and Feature Engineering <br>
      - Feature engineering <br>
      - Target feature analysis <br>
      - Model Section and Training <br>
      - Hyper parameter Tuning <br>
      - Final Model <br>
      - Results and Conclusion <br>
    
  **NOTE:** While uploading the complete notebooks, some figures donot have output due to size issues. So, while testing the script please run each cell in the '_Capstone_Project_without_output.ipynb_' notbebook individually and check out the results.
  
  I have attached the images of the missed out figures(results)in the folder **_images_**
   
- ### Capstone_Project_without_output.ipynb
  This file contains the following sections without their respective outputs: <br>
      - Libraries <br>
      - Load Data <br>
      - Data Pre-processing <br>
      - EDA and Feature Engineering <br>
      - Feature engineering <br>
      - Target feature analysis <br>
      - Model Section and Training <br>
      - Hyper parameter Tuning <br>
      - Final Model <br>
      - Results and Conclusion <br>
      
- ### train.py : 
  Final model training script.
- ### predict.py :
  Model prediction API script for local deployment.
- ### Dockerfile (inside app_deploy folder): 
  Instructions to build Docker image.
- ### app_deploy:
  Contains all the files for the docker container.
- ### requirements.txt : 
  Package dependencies and management files.
- ### Pipfile and Pipfile.lock :
  Requirements needed for this project are mentioned.
- ### heart_failure_clinical_records_dataset.csv :
  This is the dataset used for this project.
  

## 3. Virtual Environment and Packages

To ensure all scripts work fine and the libraries used during development are the ones which you use during your deployment/testing, Python venv has been used to manage virtual environment and package dependencies. Please follow the below steps to setup this up in your environment.

1. Firstly, install pip and venv if not installed: <br>
     _sudo apt install -y python-pip python-venv_
2. Create a virtual environment <br>
     _python -m venv Python3_
3. Activate the virtual environment <br>
     _. ./Python3/bin/activate_
4. Clone this repo: <br>
    _git clone_ https://github.com/amruta95/MLZoomCamp_Capstone_Project.git
   
5. Change to the directory that has the required files: <br>
     _cd_ MLZoomCamp_Capstone_Project/

6.  Install packages required for this project: <br>
     _pip install -r requirements.txt_

## 4. Deploy model as Web Service
You can deploy the trained model as a web service running inside a docker container on your local machine.

NOTE: You should have Docker installed and running on the machine where you want to perform model deployment to docker. Run the below commands to check whether docker service is running and then to see if any docker containers are running. <br>

_systemctl status docker_ <br>
_docker ps -a
_

Please follow the steps mentioned below: <br>

1. Clone this repo <br>
    _git clone_ https://github.com/amruta95/MLZoomCamp_Capstone_Project.git
2. Change to the directory that has the required files: <br>
     _cd_ MLZoomCamp_Capstone_Project/
3. Build docker image named heart-failure-prediction <br>
     _docker build -t "heart-failure-prediction" ._
4. Check docker image available. Output of below command should show the image with name heart-failure-prediction <br>
    _docker images_
    
5. Create a docker container from the image. The model prediction script as a web service will then be running inside this container. Below command will create and run a docker container named heart-td-pred (--name heart-td-pred) running as a daemon. If you want to map different port on host just change the first number), from image heart-failure-prediction. The container will be deleted if stopped or when you shutdown your machine (--rm) <br>
   _docker run --rm --name heart-td-pred -d -p 9696:9696 heart-failure-prediction_

6. Check whether docker container running. Below command should show the container in Running state and not Exited <br>
   _docker ps -a_


## 5. Deploy Model on local machine
The model can be deployed locally by that users.

To test the model deployment locally, open a terminal session on your machine, change to the direcotry where the repo is clones.Next,activate the virtual environment as explained in section 3.

Open the jupyter notebook :

_$../Capstone Project/ jupyter notebook_

Next, run the cells one by one and see the output

Another option :<br>
Open two terminals, run predict.py in one and request.py in another terminal.

