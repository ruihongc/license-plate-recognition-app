# License Plate Recognition

## Usage

### Prerequisites
- Linux
- FFmpeg
- Python ≥ 3.10
- CUDA 11/12 (Highly Recommended)

### Setup
1. Create venv: ```python -m venv venv```
2. Enter venv: ```source ./venv/bin/activate```
3. Install requirements: ```pip install -r requirements.txt```

### Data Setup
1. Create a ```guests.json``` file of ```{license_plate: guest_name}``` in the project root directory
2. Create a ```category.json``` file of ```{guest_name: guest_category}``` in the project root directory, where the category can be any tag assigned to each guest


### Run
1. Enter venv: ```source ./venv/bin/activate```
2. Start app: ```python main.py```
3. Open ```0.0.0.0:3005``` in the browser (Chromium recommended)
4. Wait for 3×```websocket connected``` 
4. Change the input source in the settings accordingly 
5. Adjust other settings to suit your use case
6. Play

## Edit
```sh
python main.py edit
```

## App endpoints
1. / or /guests (guest list)
2. /welcome (welcome screen)
3. /app (main app and control panel)

## FAQ

### Pipe RTSP stream from IP camera via FFmpeg
1. ```export VIDEO_NUMBER=1``` (replace the video number with the actual virtual camera number you want to use, use this number as the input source in the app settings)
2. ```export RTSP_ADDR=rtsp://username: password@255.255.255.255``` (replace the username , password and IP address accordingly)
3. ```sudo modprobe v4l2loopback devices=1 video_nr=$VIDEO_NUMBER exclusive_caps=1 card_label="Virtual Webcam"```
4. ```ffmpeg -rtsp_transport tcp -stream_loop -1 -re -i $RTSP_ADDR -vcodec rawvideo -threads 0 -f v4l2 /dev/video$VIDEO_NUMBER```

### Reset GPU
1. ```sudo rmmod nvidia_uvm```
2. ```sudo modprobe nvidia_uvm```

## Kill connections
Make sure you're in venv and use this command to kill connection if there are errors.
```sh
sudo kill -9 `sudo lsof -t -i:8501`
```

## Credits
- YOLO
- easyOCR
- RapidFuzz
- StreamSync UI
- All the other libraries this app is built on
- Various creators for their artwork used as static UI assets (no copyright infringement intended)
