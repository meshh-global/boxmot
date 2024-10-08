# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import time
import os
import json
import boto3
from dotenv import load_dotenv
import signal
import atexit
import torch
import multiprocessing
from multiprocessing import Process, Queue
import logging
import yaml

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import get_yolo_inferer
from tracking.detectors.stream_source import create_stream_source

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_env_vars():
    """Load environment variables from .env file."""
    load_dotenv()

def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket_name: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_path)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")
        return False
    return True

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def save_output_process(args, output_file):
    """Save output file and upload to S3 if enabled"""
    logging.info(f"Saving output file: {output_file}")
    if args.s3_upload:
        load_env_vars()
        s3_bucket = os.getenv('S3_BUCKET')
        if s3_bucket:
            s3_object_name = f"detections/{os.path.basename(output_file)}"
            if upload_to_s3(output_file, s3_bucket, s3_object_name):
                logging.info(f"Successfully uploaded {output_file} to S3 bucket {s3_bucket} as {s3_object_name}")
            else:
                logging.error(f"Failed to upload {output_file} to S3")
        else:
            logging.warning("S3_BUCKET not found in .env file. Skipping S3 upload.")

@torch.no_grad()
def run(args):
    global output_file, save_process
    
    ul_models = ['yolov8', 'yolov9', 'yolov10', 'rtdetr', 'sam']

    yolo = YOLO(
        args.yolo_model if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt',
    )

    # Use stream source if specified
    if args.stream_config:
        print(f"Stream config: {args.stream_config}")
        # Read from stream_config.yaml 
        with open(args.stream_config, 'r') as f:
            config = yaml.safe_load(f)
            source = config['stream']['url']
            print(f"Stream source: {source}")
        cap = cv2.VideoCapture(source)
    else:
        print(f"Source: {args.source}")
        source = args.source
        cap = cv2.VideoCapture(source)

    print(f"Source: {source}")

    # Create a single output file for all detections with timestamp in the filename
    start_timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.project, args.name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'detections_{start_timestamp}.txt')
    
    logging.info(f"Output file will be created at: {output_file}")
    
    last_process_time = time.time()
    fps_interval = 0.95  # 1 second interval for 1 FPS

    # Create the file even if no detections are made
    with open(output_file, 'w') as f:
        f.write("Timestamp,Number of people detected,IDs,Confidence levels\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break

        current_time = time.time()
        
        # Process frame only if 1 second has passed since the last processed frame
        if current_time - last_process_time >= fps_interval:
            last_process_time = current_time

            results = yolo.track(frame, persist=True, verbose=False, conf=args.conf, iou=args.iou, 
                                 classes=args.classes, agnostic_nms=args.agnostic_nms, 
                                 tracker=args.tracking_method)

            for r in results:
                img = r.plot()

                # Get current timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Write detections to the single output file
                if args.save_txt:
                    try:
                        with open(output_file, 'a') as f:
                            # Get detections for people (assuming class 0 is person)
                            people_detections = [box for box in r.boxes if int(box.cls) == 0]
                            num_people = len(people_detections)
                            
                            # Get IDs for each detection
                            ids = [int(box.id.item()) if box.id is not None else -1 for box in people_detections]
                            
                            # Get confidence levels for each detection
                            confidence_levels = [round(float(box.conf),2) for box in people_detections]
                            
                            # Convert IDs and confidence levels to JSON strings
                            ids_json = json.dumps(ids)
                            confidence_json = json.dumps(confidence_levels)
                            
                            f.write(f"{timestamp},{num_people},{ids_json},{confidence_json}\n")
                        logging.info(f"Wrote detection at {timestamp}: {num_people} people, IDs: {ids_json}, confidence levels: {confidence_json}")
                    except Exception as e:
                        logging.error(f"Error writing to file: {e}")

                if args.show:
                    cv2.imshow('BoxMOT', img)     
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' ') or key == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()

    logging.info(f"Tracking completed. Output file: {output_file}")
    
    # Start the save_output process
    save_process = Process(target=save_output_process, args=(args, output_file))
    save_process.start()
    save_process.join(timeout=60)  # Wait for up to 60 seconds for the process to complete

def handle_exit(signum, frame):
    """Handle exit signals."""
    logging.info("Received exit signal. Saving output...")
    try:
        if 'save_process' in globals() and save_process.is_alive():
            logging.info("Waiting for save process to complete...")
            save_process.join(timeout=60)  # Wait for up to 60 seconds
            if save_process.is_alive():
                logging.warning("Save process did not complete in time. Terminating...")
                save_process.terminate()
                save_process.join()
        else:
            logging.info("Starting save process...")
            save_process = Process(target=save_output_process, args=(opt, output_file))
            save_process.start()
            save_process.join(timeout=60)  # Wait for up to 60 seconds
            if save_process.is_alive():
                logging.warning("Save process did not complete in time. Terminating...")
                save_process.terminate()
                save_process.join()
    except NameError:
        logging.error("Output file not created yet. Exiting without saving.")
    except Exception as e:
        logging.error(f"Error during exit: {e}")
    finally:
        exit(0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--s3-upload', action='store_true',
                        help='upload output file to S3 bucket specified in .env file')
    parser.add_argument('--stream-config', type=str, default=None, 
                        help='path to YAML config file for stream source (RTP or UDP)')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_opt()
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    atexit.register(lambda: handle_exit(None, None))
    run(opt)
