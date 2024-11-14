import streamsync as ss
import asyncio
import threading
import json
import pandas as pd

plates_to_names = json.load(open("../guests.json", "r"))
names = []
data = []
for i in plates_to_names:
    if plates_to_names[i] not in names:
        names.append(plates_to_names[i])
        data.append([plates_to_names[i], ""])

df = pd.DataFrame(columns=("Name", "Present",), data=data)
counter = f"{df['Present'].value_counts()['✓'] if '✓' in df['Present'].value_counts() else 0}/{len(df)}"

initial_state = ss.init_state({
    "counter": counter,
    "guests": df,
})

# def wsthread(state):
#     with connect("ws://127.0.0.1:3005/ws") as websocket:
#         while True:
#             message = websocket.recv()
#             state["guests"] = pd.concat([state["guests"], pd.DataFrame(columns=("Name", "Present",), data=[line.split(",") for line in message.decode("latin-1").split("\n")])])

import websocket
import threading
from time import sleep
# websocket.enableTrace(True)

def get_update(state):
    state["guests"] = df
    state["counter"] = counter

def connect():
    def on_message(ws, message):
        global df, counter
        df = pd.DataFrame(columns=("Name", "Present",), data=[(",".join(line.split(",")[:-1]), line.split(",")[-1]) for line in message.split("\n")])
        v = df['Present'].value_counts()
        if '✓' in v:
            counter = f"{v['✓']}/{len(df)}"
        else:
            counter = f"0/{len(df)}"
        # state["guests"] = pd.DataFrame(columns=("Name", "Present",), data=[line.split(",") for line in message.split("\n")])

    def on_close(ws, two, three):
        sleep(1)
        connect()

    def on_open(ws):
        try:
            ws.send(f"guests")
        except:
            pass

    # global ws
    ws = websocket.WebSocketApp("ws://0.0.0.0:3005/ws", on_open = on_open, on_message = on_message, on_close = on_close)
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

connect()
# update_df = make_update_df(initial_state)
