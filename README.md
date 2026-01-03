![baxter_posed](https://github.com/user-attachments/assets/34ce0a5f-5db9-4f6b-8b4c-ef2994639d8e)

# VAP with Baxter
As part of my PhD research, I developed an algorithm we call the Variable Autonomy Planner, or VAP algorithm. It's a shared control schem for robotics which capitalizes on humans' excellent ability to intuitively scale the level of robotic versus human intervention needed to acheive a task efficiently. The algorithm sorts modular autonomous subcomponents by the degree of success of their execution (typically a joint metric of time to complete and probability of success), and allows a human to pick where in that list the cutoff is.

The exact mechanism and theory is outlined more thoroughly in the attached paper, `Sliding-Scale Autonomy for Shared Control Robotics in a Pick-and-Place Task.pdf`.

The hypothesis regarding the execution of these tasks being improved through the use of this planning algorithm were tested in a pick-and-place task using the (now obsolete) Baxter robotic manipulator. Because of the modular nature of the system, parallel manual and autonomous modes were developed for each subtask in the picking routine, and swapped in or out based on the autonomy level selection for the task being tested.

## General Outline

The overall task was for the robot manipulator to begin from an imaging pose over the workspace, target an object and estimate its pose, then drop the end effector to close range, adjust position and angle to grasp the object, complete the descent, close the gripper, and raise to the home position with the item collected.

This process overall remitted to 5 different submodules which could reasonably be toggled between manual and autonomous modes, creating 6 possible levels of autonomy under the VAP algorithm framework (from none of the 5 meing manual, to all 5 being manual)

### Manual Modules

The modules constructed for human input were based primarily on a multifunction Joystick input, with the operator dragging targeting points on an image of the workspace through the robot's camera, or rotating an angle indicator to adjust rotation. 

An example of the UI, in the position-selection mode on the left, and the orientation-selection mode on the right, is illustrated below:

<img width="1017" height="318" alt="UI_eg" src="https://github.com/user-attachments/assets/98dedc27-8276-4069-ba8f-60578210b03b" />

### Autonomous modes

The autonomous modes depend on two flavors of work- first, kinematic approach and selection, which is based on calculating image position parameters, and partial motion functions, which execute motions to given selected target poses.

The former is primarily based on two image processing routines, one which uses a histogram-based feature vector to identify objects by type, and then use a background-subtraction method to identify a region of interest and extract object center coordinates, and attempt to calculate the minimum-width axis for grasping angle. Again, this is explained more thoroughly in the attached paper, but an illustration of the algorithm's output is shown below:

<img width="1023" height="317" alt="obj_ID_track" src="https://github.com/user-attachments/assets/8e8fb785-46df-483d-89f6-f4a055029504" />

Here, we can see the identified object centers, orientation crosshairs, and the classification results of the feature vector check.

The second image processing mode is for the fine-adjust mechanism, which runs when the gripper is close to the object. It uses a local assumption that the camera is close enough to the object that the area around the middle 1/3 rectangle is background, and segments the image by taking a difference between the average color vector in the 'border' region from that inside the middle area. The extracted blob is then analyzed for COM and orientation much the same as in the other mode.

## Results

The results with the algorithm were promising, with substantial performance improvements noted within the sliding scale autonomy choice over the fixed autonomy level results. Users were able to accurately predict nd select the optimal autonomy level for the task after a very short acclimatization period. Further, observations in the trends of performance latr informed mathematical analysis of a more sophisticated version of the algorithm (later called VAP+) to accomodate non-sequential tasks such as this one.

Experiments were conducted under conditions of escalating complexity: in which the workspace included one item, a set of spaced out items, and a set of items cluttered closely together. (As indicated in the sample image folders 'space sample' and `seg sample`, respectively)

The results for the user trials, indicating the coherence between selection and optimization are indicated here:

<img width="1855" height="834" alt="Data key Var auto paper" src="https://github.com/user-attachments/assets/2c9b1d36-cb2d-4756-aeb0-bc7a17d3377f" />

Additionally, the attached videos comcisely demonstrate the system's performance, in both autonomous and manual execuation cases.



