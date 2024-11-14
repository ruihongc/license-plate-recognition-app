import streamsync as ss
import asyncio
import threading
import json
import pandas as pd

message = " "
ticks = 0
categories=" "
with open('../category.json') as json_data:
    category_dict = json.load(json_data)

initial_state = ss.init_state({
    "message": message,
    "categories":categories
})

# def wsthread(state):
#     with connect("ws://0.0.0.0:3005/ws") as websocket:
#         while True:
#             message = websocket.recv()
#             state["guests"] = pd.concat([state["guests"], pd.DataFrame(columns=("Name", "Present",), data=[line.split(",") for line in message.decode("latin-1").split("\n")])])

import websocket
import threading
from time import sleep
# websocket.enableTrace(True)

def get_update(state):
    global message, categories, ticks
    state["message"] = message
    state["categories"]=categories
    ticks += 1
    if ticks >= 100:
        ticks = 0
        message = " "
        categories = " "

def connect():
    def on_message(ws, name):
        global message, ticks, categories
        try:
            if category_dict[name]!='':
                categories = category_dict[name]
            else:
                categories = " "
        except:
            categories = " "
            print("name not found in category list")
        message = name
        ticks = 0
        # state["guests"] = pd.DataFrame(columns=("Name", "Present",), data=[line.split(",") for line in message.split("\n")])

    def on_close(ws, two, three):
        sleep(1)
        connect()

    def on_open(ws):
        try:
            ws.send(f"welcome")
        except:
            pass
    # global ws
    ws = websocket.WebSocketApp("ws://0.0.0.0:3005/ws", on_open = on_open, on_message = on_message, on_close = on_close)
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

connect()
# update_df = make_update_df(initial_state)
