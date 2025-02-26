from ultralytics import YOLO
from mjpeg_streamer import MjpegServer, Stream
from processplates import reordered_plates, double_replace
from datetime import datetime
import streamsync as ss
import pandas as pd
import cv2
import easyocr
import torch
import numpy as np
import copy
import rapidfuzz
import requests
import threading
import json

plates_to_names = json.load(open("../guests.json", "r"))
plates_to_names = reordered_plates(plates_to_names)
print(plates_to_names)

guests = {}
attendance = {}
plates_list = []
names_list = []

for i in plates_to_names:
    guests[plates_to_names[i]] = plates_to_names[i]
    attendance[plates_to_names[i]] = ""
    plates_list.append(i)
    names_list.append(plates_to_names[i])

# stream = Stream("stream", size=(width, height), quality=50, fps=fps)
stream = Stream("stream", quality=100, fps=20)
server = MjpegServer("0.0.0.0", 8501)
server.add_stream(stream)
server.start()

placeholder = cv2.imdecode(np.frombuffer(open("./static/placeholder.jpg", "rb").read(), np.uint8), cv2.IMREAD_COLOR)
stream.set_frame(placeholder)

model_path = "../models/"
folder_path = "../licenses_plates_imgs_detected/"
license_plate_detector = YOLO(model_path + "license_plate_detector.pt")
# mapping_table = str.maketrans('', '', string.punctuation + " ")

def read_license_plate(license_plate_crop, params, sorting_tolerance, min_text_percentage): #, img):
    scores = 0
    detections = reader.readtext(license_plate_crop, **params) # decoder='beamsearch', beamWidth=64, batch_size=256, text_threshold=0.7, low_text=0.4) #, **params)# decoder=params["decoder"], allowlist=params["allowlist"], beamWidth=params["beamWidth"], batch_size=params["batch_size"], text_threshold=params["text_threshold"], low_text=params["low_text"])
    # width = img.shape[1]
    # height = img.shape[0]
    
    if detections == [] :
        return None, None

    rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]
    plate = []
    sorted_detections = sorted( detections, key=lambda result: (int(result[0][0][1]/license_plate_crop.shape[1]/sorting_tolerance), result[0][0][0]) )

    # print(sorted_detections)
    for (bbox, text, prob) in sorted_detections:
        # length = np.subtract(result[0][1][0], result[0][0][0])
        length = np.sum(np.subtract(bbox[1], bbox[0]))
        # height = np.subtract(result[0][2][1], result[0][1][1])
        height = np.sum(np.subtract(bbox[2], bbox[1]))
        if length*height / rectangle_size > min_text_percentage:
            scores += prob
            plate.append(text.upper())
            # plate.append(text.upper().translate(mapping_table))
    
    if len(plate) != 0 : 
        return "".join(plate), scores/len(plate)
    else :
        return "".join(plate), 0

"""def update_image(state, image):
    state["image"] = image"""

def play(state):
    try:
        params = copy.deepcopy(state["params"])
        easyocr_params = {}
        easyocr_params["decoder"] = str(params["easyocr"]["decoder"])
        easyocr_params["allowlist"] = str(params["easyocr"]["allowlist"])
        easyocr_params["beamWidth"] = int(params["easyocr"]["beamWidth"])
        easyocr_params["batch_size"] = int(params["easyocr"]["batch_size"])
        easyocr_params["text_threshold"] = float(params["easyocr"]["text_threshold"])
        easyocr_params["low_text"] = float(params["easyocr"]["low_text"])
        easyocr_params["link_threshold"] = float(params["easyocr"]["link_threshold"])
        sorting_tolerance = float(params["sorting_tolerance"])
        min_text_percentage = float(params["min_text_percentage"])
        expand_x = int(params["expand_x"])
        expand_y = int(params["expand_y"])
        res_x = int(params["res_x"])
        res_y = int(params["res_y"])
        similarity = float(params["similarity"])
        send = params["send"]["0"] == "yes"
        guests_list = params["guests_list"]["0"] == "yes"
        url = str(params["url"])
        location = str(params["location"])
        font_scale = float(params["font_scale"])
        # state.add_notification("info", "Using parameters", str(params))
    except Exception as e:
        state.add_notification("error", "Error processing parameters", str(e))
        return
    if state["src"].isdigit():
        src = int(state["src"])
    else:
        src = state["src"]
    vc = cv2.VideoCapture(src)
    try:
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)
        vc.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vc.get(cv2.CAP_PROP_FPS)
        # streaming_process = start_streaming(width, height, fps)
        state["running"] = "yes"
        # state["image"] = "http://0.0.0.0:8501/stream"
        while state["running"] == "yes":
            _, img = vc.read()
            if img is None:
                state.add_notification("error", "Cannot open source", "Failed to open video source.")
                state["running"] = "no"
                break
            img_to_an = img.copy()
            license_detections = license_plate_detector(img_to_an, verbose=False)[0]
            if len(license_detections.boxes.cls.tolist()) != 0 :
                dets = []
                for license_plate in license_detections.boxes.data.tolist() :
                    try:
                        x1, y1, x2, y2, score, class_id = license_plate
                        license_plate_crop = img[int(y1) - expand_y:int(y2) + expand_y, int(x1) - expand_x: int(x2) + expand_x, :]
                        # license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, easyocr_params, sorting_tolerance, min_text_percentage) #, img)

                        if license_plate_text:
                            license_plate_text = str(license_plate_text)
                            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            # cv2.putText(img,
                            #             license_plate_text,
                            #             (int(x1 + 10), int(y1) - 10),
                            #             cv2.FONT_HERSHEY_SIMPLEX,
                            #             1, (0, 0, 0), 2)
                            dets.append((str(datetime.now()), license_plate_text, license_plate_text_score,)) # x1, y1, x2, y2, score, class_id))
                            if guests_list:
                                res = rapidfuzz.process.extractOne(double_replace(license_plate_text), plates_list)
                                # state.add_notification("info", "res", str(res) + "\n" + names_list[res[2]])
                                name = names_list[res[2]]
                                if (res[1] > similarity) and (name not in state["present_guests"]):
                                    state["present_guests"] = sorted(list(state["present_guests"]) + [name])
                                    attendance[name] = "✓"
                                    state.add_notification("info", "VIP has arrived", f"{name} has arrived.")
                                    cv2.rectangle(img, (int(x1), int(y1) - int(40*font_scale)), (int(x2) + int(len(name)*10*font_scale) + 20, int(y1)), (255, 255, 255), cv2.FILLED)
                                    cv2.putText(img,
                                            name,
                                            (int(x1 + 10), int(y1) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            font_scale, (0, 0, 0), 2)
                                    if send:
                                        ws.send(f"{name},✓")
                                        send_data(url, name, location)
                                    # send data
                                else:
                                    cv2.rectangle(img, (int(x1), int(y1) - 40), (int(x2) + len(license_plate_text)*10 + 20, int(y1)), (255, 255, 255), cv2.FILLED)
                                    cv2.putText(img,
                                        license_plate_text,
                                        (int(x1 + 10), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 0), 2)
                            else:
                                cv2.rectangle(img, (int(x1), int(y1) - 40), (int(x2) + len(license_plate_text)*10 + 20, int(y1)), (255, 255, 255), cv2.FILLED)
                                cv2.putText(img,
                                        license_plate_text,
                                        (int(x1 + 10), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 0), 2)

                        # streaming_process.stdin.write(img.tobytes())
                        # update_image(state, cv2.imencode(".bmp", img)[1].tobytes())
                    except:
                        pass
                if dets: state["results"] = pd.concat([state["results"], pd.DataFrame(columns=("Time", "License Plate Text", "Text Score",), data=dets)], axis=0) # "x1", "y1", "x2", "y2", "score", "class_id"), data=dets)], axis=0)
            stream.set_frame(img)
    except:
        pass
    finally:
        stream.set_frame(placeholder)
        # state["image"] = open("./static/placeholder.jpg", "rb").read()
    # streaming_process.stdin.close()
    # streaming_process.wait()
        # try:
        #     server.stop()
        # except:
        #     pass
        vc.release()
        state.add_notification("info", "Terminated", "Licence plate detection system terminated.")
        state["running"] = "no"

def request_task(url, name, location):
    try:
        requests.post(url, data={"name": " ".join(name.split(" ")[1:]), "location": location})
    except:
        pass

def send_data(url, name, location):
    # pool.apply_async(requests.get, (url,))
    threading.Thread(target=request_task, args=(url, name, location)).start()

def cancel(state):
    stream.set_frame(placeholder)
    # state["image"] = open("./static/placeholder.jpg", "rb").read()
    state["running"] = "no"
    state.add_notification("info", "Stopping", "Stopping licence plate detection system...")

def clear_results(state):
    state["results"] = pd.DataFrame(columns=("Time", "License Plate Text", "Text Score",), data=[(str(datetime.min), "SAMPLE", 1.0)])# "X1", "Y1", "X2", "Y2", "Score", "Class ID"), data=[[None]*8]),

def mark_present(state):
    for i in attendance:
        if (i in state["present_guests"]):
            if (attendance[i] != "✓"):
                if (state["params"]["send"]["0"] == "yes"):
                    send_data(str(state["params"]["url"]), i, str(state["params"]["location"]))
                attendance[i] = "✓"
                ws.send(f"{i},✓")
        elif (attendance[i] != ""):
            attendance[i] = ""
            ws.send(f"{i},")
    

def ui_guests_list(state):
    state["params"]["guests_list"] = {
        "0": "yes",
        "1": "no",
        "T": True,
    }

def ui_guests_list_not(state):
    state["params"]["guests_list"] = {
        "0": "no",
        "1": "yes",
        "T": False,
    }

def ui_send(state):
    state["params"]["send"] = {
        "0": "yes",
        "1": "no",
    }

def ui_send_not(state):
    state["params"]["send"] = {
        "0": "no",
        "1": "yes",
    }

initial_state = ss.init_state({
    "image": "http://0.0.0.0:8501/stream", # open("./static/placeholder.jpg", "rb").read(),
    "running": "no",
    "results": pd.DataFrame(columns=("Time", "License Plate Text", "Text Score",), data=[(str(datetime.min), "SAMPLE", 1.0)]), # "X1", "Y1", "X2", "Y2", "Score", "Class ID"), data=[[None]*8]),
    "src": "0",
    "params": {
        "easyocr": {
            "decoder": 'beamsearch',
            "allowlist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "beamWidth": "8",
            "batch_size": "64",
            "text_threshold": "0.70",
            "low_text": "0.40",
            "link_threshold": "0.40",
        },
        "sorting_tolerance": "0.33",
        "min_text_percentage": "0.14", 
        "expand_x": "0",
        "expand_y": "0",
        "res_x": "1920",
        "res_y": "1080",
        "font_scale": "5.0",
        "similarity": "90",
        "guests_list": {
            "0": "yes",
            "1": "no",
            "T": True,
        },
        "send": {
            "0": "no",
            "1": "yes",
        },
        "url": "http://0.0.0.0:8080/detections",
        "location": "Entrance",
    },
    "guests": guests,
    "present_guests": [],
})

import websocket
import threading
from time import sleep
websocket.enableTrace(False)

def on_message(ws, message):
    pass

def on_close(ws, two, three):
    sleep(1)
    connect(None)

def connect(state):
    global ws
    ws = websocket.WebSocketApp("ws://0.0.0.0:3005/ws", on_open = on_open, on_message = on_message, on_close = on_close)
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    try:
        ws.send(f"app")
    except:
        pass

def on_open(ws):
    try:
        ws.send(f"app")
    except:
        pass

connect(None)

if torch.cuda.is_available():
    initial_state.add_notification("success", "CUDA is available", "CUDA is available. Using the GPU mode.")
    # coco_model.to('cuda')
    license_plate_detector.to('cuda')
    reader = easyocr.Reader(['en'], model_storage_directory=model_path, gpu=True)
else:
    initial_state.add_notification("warning", "CUDA is unavailable", "CUDA is unavailable. Using the CPU mode.")
    reader = easyocr.Reader(['en'], model_storage_directory=model_path, gpu=False)

initial_state.add_notification("success", "Ready", "Licence plate detection system loaded and ready to use.")
