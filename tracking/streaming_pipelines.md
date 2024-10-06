
## Streaming sender (UDP)

```bash
gst-launch-1.0 avfvideosrc device-index=0 ! \
video/x-raw,width=640,height=480,framerate=30/1 ! queue ! \
videorate max-rate=1 ! videoconvert ! queue ! \
x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! queue ! \
rtph264pay config-interval=1 pt=96 ! \
udpsink host=38.128.233.122 port=5000

```

## Streaming receiver (UDP)

```bash
gst-launch-1.0 udpsrc port=5000 !  application/x-rtp, media=video, encoding-name=H264  ! rtpjitterbuffer latency=100  !  rtph264depay ! queue !  h264parse ! avdec_h264 ! videoconvert ! videoscale ! gtksink -e
```