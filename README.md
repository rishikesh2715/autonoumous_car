# Smart Sidewalk Navigation: Autonomous Toy Car  

*ECE Project Lab 2 - Texas Tech University*

## Overview

This project showcases an autonomous toy car with advanced navigation capabilities using the YOLOv8 object detection framework. The car can:

* Detect and follow sidewalks.
* Detect humans and stop to prevent collisions.
* Detect traffic cones and respond with appropriate speed adjustments (slow down, full speed, or stop). 

## Hardware Requirements

* **Toy Car Base:** (Specify brand and model)
* **Raspberry Pi:** (Model 5 or better yet A Nvidia Jetson Based SBC)
* **Camera Module:** (Logitech C920)
* **Motor Driver:** (Specify mode)
* **Servo Motor:** (Specify model)
* **Power Supply:** (Specify type)
* **Jumper Wires, Breadboard** 

## Software Requirements

* **Operating System:** Raspbian OS 
* **Python 3.11** 
* **OpenCV**
* **YOLOv8** ([Installation instructions at: https://github.com/ultralytics/ultralytics])
* **Additional Libraries:**
   * `numpy`
   * `torch`
   * `time`
   * `RPi.GPIO` (If using Raspberry Pi)

## Project Structure

* **src/**
    * **yolo_sidewalk_detection.py**
    * **yolo_human_detection.py**
    * **yolo_traffic_cone_detection.py**
    * **navigation.py** 
* **models/**
    * **sidewalk_model.pt** 
    * **human_model.pt** 
    * **traffic_cone_model.pt**
* **data/** (Optional, if using a custom dataset)

## Setting Up the Project

1. **Hardware Assembly:** (Provide brief instructions)
2. **Software Installation:** (List installation steps)
3. **Download/Train Models:** (Describe where to get models or how to train)
4. **Clone the Repository:** (Provide the `git clone` command) 

## Running the Car

1. **Setup Test Environment:** (Quick description of the test setup)
2. **Run Script:** `python src/navigation.py`

## Training Your Own YOLOv8 Models

(Provide instructions or a link to instructions)

## Demo Video

[Link to a demo video of the car in action]

## Results

(Summarize performance metrics and observations)

## Troubleshooting

(Add common issues and potential solutions)

## Contributing

(Outline guidelines for how others can contribute)

## Acknowledgments

(List any resources, tutorials, or people that helped) 
