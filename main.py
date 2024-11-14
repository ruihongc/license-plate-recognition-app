import uvicorn
import streamsync.serve
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
import json
import os
import sys

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    async with sub_asgi_app.router.lifespan_context(app):
        async with sub_asgi_guests.router.lifespan_context(app):
            async with sub_asgi_welcome.router.lifespan_context(app):
                yield


plates_to_names = json.load(open("guests.json", "r"))
names = []
names_index = {}

for i in plates_to_names:
    if plates_to_names[i] not in names_index:
        names_index[plates_to_names[i]] = len(names)
        # names.append({
        #     "name": plates_to_names[i],
        #     # "", # Time
        #     # "", # Updated based on
        #     # "", # Updated by
        #     "present": False, # Present
        # })
        names.append([plates_to_names[i], ""])

root_asgi_app = FastAPI(lifespan=lifespan_context)
# if root_asgi_app.router:
#    root_asgi_app.router.redirect_slashes = False
# static_route = root_asgi_app.routes[-1]
# root_asgi_app.router.routes = root_asgi_app.routes[:-1]

@root_asgi_app.get("/")
async def init():
    # return Response("""
    #     <h1>Welcome</h1>
    # """)
    return RedirectResponse("/guests")

class ConnectionManager:
    def __init__(self):
        # self.active_connections: list[WebSocket] = []
        # self.conn_type: list[str] = []
        # self.n: int = 0
        self.active_connections = {}

    # async def connect(self, websocket: WebSocket):
    #     await websocket.accept()
    #     self.active_connections[websocket] = ""
    #     # self.active_connections.append(websocket)

    def connect(self, websocket: WebSocket, conn: str):
        self.active_connections[websocket] = conn

    def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket]
        # self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, group: str):
        for connection in self.active_connections:
            if self.active_connections[connection] == group:
                await connection.send_text(message)

manager = ConnectionManager()

@root_asgi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global names
    await websocket.accept()
    # await manager.connect(websocket)
    conn = await websocket.receive_text()
    manager.connect(websocket, conn)
    # print(manager.active_connections)
    if conn == "guests":
        await manager.send_personal_message("\n".join(",".join(line) for line in names), websocket)
    try:
        while True:
            data = (await websocket.receive_text()).split(",")
            names[names_index[",".join(data[:-1])]][1] = data[-1]
            # names = [line.split(",") for line in data.split("\n")]
            await manager.broadcast("\n".join(",".join(line) for line in names), "guests")
            if data[1]:
                await manager.broadcast(",".join(data[:-1]), "welcome")
                
            # await manager.broadcast(data)
            # await manager.send_personal_message(f"You wrote: {data}", websocket)
            # await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # await manager.broadcast(f"Client #{client_id} left the chat")

mode = sys.argv[1] if len(sys.argv) > 1 else "run"
sub_asgi_app = streamsync.serve.get_asgi_app(os.path.join(os.getcwd(), "app"), mode)
sub_asgi_guests = streamsync.serve.get_asgi_app(os.path.join(os.getcwd(), "guests"), mode)
sub_asgi_welcome = streamsync.serve.get_asgi_app(os.path.join(os.getcwd(), "welcome"), mode)
# if sub_asgi_guests.router:
#    sub_asgi_guests.router.redirect_slashes = False

# sub_asgi_app_2 = streamsync.serve.get_asgi_app("../app2", "run")

root_asgi_app.mount("/app", sub_asgi_app)
root_asgi_app.mount("/guests", sub_asgi_guests)
root_asgi_app.mount("/welcome", sub_asgi_welcome)
# root_asgi_app.router.routes.append(static_route)
# root_asgi_app.mount("/app2", sub_asgi_app_2)

# @root_asgi_app.get("/")
# async def init():
#     return Response("""
#         <h1>Welcome</h1>
#     """)
    # return RedirectResponse("/guests")

if __name__ == "__main__":
    uvicorn.run(root_asgi_app,
        # host="127.0.0.1",
        host="0.0.0.0",
        port=3005,
        log_level="warning",
        ws_max_size=streamsync.serve.MAX_WEBSOCKET_MESSAGE_SIZE)
