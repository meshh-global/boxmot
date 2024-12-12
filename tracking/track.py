# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import atexit
from multiprocessing import Process, set_start_method
import signal
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import time
from datetime import datetime, timezone
import json
import logging
from dotenv import load_dotenv
import polars as pl

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for data storage
tracking_data = {
    'datetime_utc': [],
    'elapsed_time(s)': [],
    'total_detections': [],
    'detected_ids': [],
    'confidence_values': []
}

def save_parquet_file(data, parquet_file):
    """Save current data to parquet file"""
    try:
        if data['datetime_utc']:  # Only save if there's data
            df = pl.DataFrame({
                'datetime_utc': data['datetime_utc'],
                'elapsed_time(s)': data['elapsed_time(s)'],
                'total_detections': data['total_detections'],
                'detected_ids': data['detected_ids'],
                'confidence_values': data['confidence_values']
            })
            df.write_parquet(parquet_file)
            logging.info(f"Successfully saved parquet file: {parquet_file}")
    except Exception as e:
        logging.error(f"Error saving parquet file: {e}")

def load_env_vars():
    """Load environment variables from .env file."""
    load_dotenv()

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


def save_output_process(args, txt_file, parquet_file):
    """Save output files and upload to S3 if enabled"""
    logging.info(f"Saving output files: {txt_file} and {parquet_file}")
    
    # Save parquet file with current data
    save_parquet_file(tracking_data, parquet_file)
    
    if args.s3_upload:
        load_env_vars()
        s3_bucket = os.getenv('S3_BUCKET')
        if s3_bucket:
            # Upload txt file
            txt_object_name = f"detections/{txt_file.name}"
            if upload_to_s3(txt_file, s3_bucket, txt_object_name):
                logging.info(f"Successfully uploaded {txt_file} to S3 bucket {s3_bucket} as {txt_object_name}")
            else:
                logging.error(f"Failed to upload {txt_file} to S3")
            
            # Upload parquet file
            parquet_object_name = f"detections/{parquet_file.name}"
            if upload_to_s3(parquet_file, s3_bucket, parquet_object_name):
                logging.info(f"Successfully uploaded {parquet_file} to S3 bucket {s3_bucket} as {parquet_object_name}")
            else:
                logging.error(f"Failed to upload {parquet_file} to S3")
        else:
            logging.warning("S3_BUCKET not found in .env file. Skipping S3 upload.")


@torch.no_grad()
def run(args):
    
    global output_file, parquet_file, save_process, tracking_data

    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)

    yolo = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(args.yolo_model):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    yolo.predictor.custom_args = args

    # Create detection results file with timestamp in name
    save_path = Path(args.project) / args.name
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = save_path / f'detection_results_{timestamp}.txt'
    parquet_file = save_path / f'detection_results_{timestamp}.parquet'

    logging.info(f"Project root: {Path(args.project)}")
    logging.info(f"Output files will be created at: {output_file} and {parquet_file}")

    # Reset tracking data
    tracking_data = {
        'datetime_utc': [],
        'elapsed_time(s)': [],
        'total_detections': [],
        'detected_ids': [],
        'confidence_values': []
    }

    # Write header to results file
    with open(output_file, 'w') as f:
        f.write("datetime_utc, elapsed_time(s), total_detections, detected_ids, confidence_values\n")

    start_time = time.time()
    frame_count = 0
    for r in results:
        frame_count += 1
        # Get current timestamp
        elapsed_time = time.time() - start_time
        current_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        try:
            people_detections = [box for box in r.boxes if int(box.cls) == 0]
            num_people = len(people_detections)
            
            # Get IDs for each detection
            ids = [int(box.id.item()) if box.id is not None else -1 for box in people_detections]
            
            # Get confidence levels for each detection
            confidence_levels = [round(float(box.conf),2) for box in people_detections]
            
            # Convert IDs and confidence levels to JSON strings for txt file
            people_ids = json.dumps(ids)
            confidence_values_list = json.dumps(confidence_levels)

            if num_people > 0:
                # Save results to txt file
                with open(output_file, 'a') as f:
                    f.write(f"{current_utc}, {elapsed_time:.3f}, {num_people}, {people_ids}, {confidence_values_list}\n")
                
                # Append data for parquet file
                tracking_data['datetime_utc'].append(current_utc)
                tracking_data['elapsed_time(s)'].append(round(elapsed_time, 3))
                tracking_data['total_detections'].append(num_people)
                tracking_data['detected_ids'].append(ids)
                tracking_data['confidence_values'].append(confidence_levels)

                # Save parquet file periodically (every 100 frames)
                if frame_count % 100 == 0:
                    save_parquet_file(tracking_data, parquet_file)

        except Exception as e:
                logging.error(f"Error writing to file: {e}")
    
        if args.show is True:
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
    
    # Final save of parquet file
    save_parquet_file(tracking_data, parquet_file)
    
    logging.info(f"Tracking completed. Output files: {output_file} and {parquet_file}")


def handle_exit(signum, frame):
    """Handle exit signals."""
    logging.info(f"Received exit signal {signum}...")
    try:
        if 'output_file' in globals() and 'parquet_file' in globals():  # Only proceed if output files exist
            # Save parquet file immediately with current data
            save_parquet_file(tracking_data, parquet_file)
            
            if 'save_process' not in globals() or not save_process.is_alive():
                logging.info("Starting save process...")
                save_process = Process(target=save_output_process, args=(opt, output_file, parquet_file))
                save_process.start()
                save_process.join(timeout=60)  # Wait for up to 60 seconds
                if save_process.is_alive():
                    logging.warning("Save process did not complete in time. Terminating...")
                    save_process.terminate()
                    save_process.join()
            else:
                logging.info("Save process already running...")
                save_process.join(timeout=60)  # Wait for up to 60 seconds
                if save_process.is_alive():
                    logging.warning("Save process did not complete in time. Terminating...")
                    save_process.terminate()
                    save_process.join()
    except NameError:
        logging.error("Output files not created yet. Exiting without saving.")
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
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None,
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

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    set_start_method('spawn')
    opt = parse_opt()
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    atexit.register(lambda: handle_exit(None, None))
    run(opt)
