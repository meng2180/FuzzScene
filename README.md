# Semantic-Guided Scene Fuzzing for Virtual Testing of Autonomous Driving Systems
This repository is for providing the basic data and code for our approach

<div align=center><img width="85%" height="80%" src="https://anonymous.4open.science/r/FuzzScene-3133/gif/ArrowedAngle_ori.gif"/></div>
<div align=center><img width="85%" height="80%" src="https://anonymous.4open.science/r/FuzzScene-3133/gif/ArrowedAngle_mut.gif"/></div>

## Repository Structure

+ FuzzScene

  + gif

    this subdirectory saves two gif pictures of simulation, the  first one with blue arrow represents the predict angle of origin scene, and the second one with red angle represents the predict angle of scene after mutation, which shows on the README file

  + code
  
    this subdirectory includes code, basic data, trained models and some scripts for demonstrating our experiments.
  
    + the `EA` folder saves the main code of our algorithms
    + the `env` folder saves two anaconda env used in experiments
    + the `sampling_vgg` folder saves the VGG-Net models for our sampling algorithm
    + the `seed_pool` folder saves the six basic Open Scenario scenes
    + the `trained_models` folder saves the four trained driving models
    + the `Violated images` folder saves the error images found by our algorithm, divided by four models
    + the `scenario_runner` folder saves the scenario_runner files which helps run the simulation
  


  + README.md


## Environment

+ Python Env
  + we use **Anaconda** for switching to different experimental environments, because of the difference between **Carla** running environment (python 3.7) and the **Driving Model** environment (python 3.6)
  + we named the **Carla** anaconda environment **carla** and **Driving Model** environment **dave**
  + two anaconda env have been saved in the `env` folder
  
+ Carla & Scenario Runner Env
  + in order to conduct our experiment, we use **Carla Simulator-0.9.13** as our autonomous driving simulator, which can be found at https://github.com/carla-simulator/carla
  + in order to build simulation scenarios, we use **Scenario Runner-0.9.13** which can be found at https://github.com/carla-simulator/scenario_runner. The files **has already been** included in our code

## Files Description

Because the trained model files in our code is too large, we provide Google Drive link to obtain these files in path *FuzzScene/code/sampling_vgg/* and *FuzzScene/code/trained_models/*

+ You can simply get these files through the link below
  + https://drive.google.com/drive/folders/18LgyPcKJOIQ-_ujr9vnNLoH0hW4I9RWo?usp=sharing
+ please make sure the files are in the right path consistent with the Google Drive

## Code Usage

We have made three scripts for conducting our experiments.

First, change the anaconda path code in `FuzzScene/code/EA/ga_sim.sh` to your anaconda path, and make sure the `CARLA_0.9.13` folder and the `FuzzScene` folder are in the same folder

then change the current anaconda environment to dave and get into the EA folder.

```bash
conda activate dave
cd FuzzScene/code/EA
```

Then you can run the **first** script to set basic parameter for experiments.

```python
python set_para.py 1 1 1
```

The first argument value can be `'1'` to `'4'`, which represents the four autonomous driving models. The second argument value can be `'0'` or `'1'`, which represents whether conduct the sampling step of the genetic algorithm or not. The last argument value can be `'1'` to `'3'`, which represents the three kinds of fitness function of the genetic algorithm. The error image found by our algorithm will be moved to the `Violated images` folder

The **second** script is for the main function of our approach, which includes initialization, all steps of our algorithm and the data collection.

```python
python fuzz_ga.py
```

The **Third** script is for moving data collected before to specified location and clear unnecessary data.
```python
python rename.py
```

Data will be moved to the folder **ga_output**, including the **file of errors** found by algorithm, which will be renamed by model name, whether conduct sampling step, and the fitness function been chose,  the **entropy** file in entropy folder, and the **r_list** file for details of experiments in r_list folder

