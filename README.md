
Project by Luca Benzi (168206@studenti.unimore.it)

# AutonomousDrive
Implementation of model for autonomous drive

BEFORE THE EXECUTION:
Make sure you have installed the libraries present in requirements.txt

Files:
- object_tracker.py: main file to execute.
- models.py: contains the network model for the detection
- moving_object.py: contains the MovingObject class
- sort.py: The SORT algorithm: Original code on: https://github.com/abewley/sort
- evaluation.py: Contains the code for the evaluation of segmentation.

Directories:
- config: Contains the configuration file of yolov3. https://github.com/pjreddie/darknet
- eval: contains the prediction frames and its respective hand-labeled groundtruth frame for evaluation purposes (see evaluation.py)
- pretrained_models: contains the pretrained DeepLabV3 and ResNet used in segmentation.
- segmentation_models: contains the implementation of DeepLabV3 architecture. Original Code on: https://github.com/fregu856/deeplabv3
- utils: contains utilities for the code.
- videos: directory where dataset videos should be put.
	  It contains 2 Dr(eye)ve example: 02 and 11 with its corrispective output (30 seconds each).


HOW TO EXECUTE:
python object_tracker.py [-p <path-to-video-directory>][-fps <fps>][-mf <minframes>][-sb <seconds-to-wait>][-se <seconds-to-end>]

-p path: optional. Default 'videos/02/'
	specify the path of the video directory (example: videos/02/).
	NB: video_garmin.avi and speed_course_coord.txt are required in the selected directory, otherwise the program won't work.
	This names and extensions of the files are mandatory (they are the same as provided in DR(eye)VE dataset).

-fps: optional. Default = 25
	Frame per seconds of the video in input.

-mf: optional. Default = 10
	Number of frames for the speed sampling.

-sb: optional. Default = 0
	Seconds of the video in which starting the execution (SHOULD BE LOWER THAN 300 and LOWER THAN se).

-se: optional. Default = 300
	Seconds of the video in which stopping the execution (SHOULD BE LOWER THAN 300 and HIGHER THAN sb).

The outputs of our network are located in the same directory specified by -p argument:
- video_garmin-det.avi is the output of the detection phase with speed evaluation of moving objects and position evaluation of objects on the road
- video_garmin-seg.avi is the output of the segmentation phase.

For all the informations please refers to the report we provide and for any question feel free to contact me in my email.
