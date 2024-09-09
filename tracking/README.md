## People counting module

The people counting module is a computer vision pipeline consisting of multiple stages:

1. Object Detection: This stage uses a deep learning model to detect and localize objects, in this case, people, in the input frames. The detected objects are assigned unique IDs.

2. Object Tracking: Once the objects are detected, this stage tracks their movement across frames using various algorithms, such as Kalman filters or Hungarian algorithm. The goal is to maintain the identity of each object throughout the video.

3. Re-identification: In this stage, the tracked objects are re-identified to handle occlusions or temporary disappearances. This is done by comparing appearance features extracted from the objects using deep learning models.

4. Output Generation: Finally, the tracking script generates an output file that contains the timestamp, the number of people detected, and confidence levels for each person. This information can be used for further analysis or monitoring.

For more details and the complete implementation, you can refer to the parent repository: [boxmot](https://github.com/mikel-brostrom/boxmot)


To use the tracking script:

1. Navigate to the directory containing the `track.py` file.
2. Run the script with the following command:

```
python tracking/track.py --source 0 --classes 0 --save-txt --show --reid-model weights/osnet_x0_25_msmt17.pt-clear --name detections
```

On a RaspberryPi (tested with RPi 5 with 8GB RAM), use the ReID model below to lighten the CPU load:
```
python tracking/track.py --source 0 --classes 0 --save-txt --show --reid-model osnet_x0_25_market1501.pt --name detections
```

The above assumes a camera or video source is connected and indexed as 0.

The script will create an output file named `detections_YYYYMMDD_HHMMSS.txt` in the `runs/track/detections` directory. `detections` is the project name and dir. The default project is `exp`.

Each line in the output file will contain:
- The timestamp
- The number of people detected
- A JSON-encoded list of confidence levels for each detected person

For example, a line in the output file might look like this:
```
2023-05-10 14:30:45,3,[0.92, 0.87, 0.95]
```

This indicates that at 14:30:45, three people were detected with confidence levels of 0.92, 0.87, and 0.95 respectively.

The script will also print information about each detection to the console, which can be useful for monitoring the tracking process in real-time.

