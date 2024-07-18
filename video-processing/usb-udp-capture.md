# USB/UDP Camera Streaming Capture, Viewing, and Frame Capturing

## 1.0 USB/UDP Camera Streaming Capture, Viewing, and Frame Capturing

### 1.1 - ffmpeg Installation

FFmpeg is a free and open-source software project that consists of a suite of libraries and programs for handling video, audio, and other multimedia files and streams. At its core, FFmpeg includes the command-line tool `ffmpeg`, which is designed for processing video and audio files. It is widely used for format transcoding, basic editing, video scaling, video post-production effects, and ensuring standards compliance. FFmpeg's libraries, such as `libavcodec` for audio/video codec support and `libavformat` for multiplexing and demultiplexing, are utilized by many software projects, including media players like VLC and platforms such as YouTube. FFmpeg is written primarily in the C programming language. 

- Install ffmpeg-full using chocolately in Windows: 
  - `choco install ffmpeg-full -y`

#### References

- [ffmpeg oveview](https://ffmpeg.org/about.html)
- [Other ffmpeg installation instructions](https://ffmpeg.org/download.html)
- [ffmpeg documentation](https://ffmpeg.org/documentation.html)

### 1.2 - Video Streaming

This process involves finding the USB cameras identified in your system and locating the camera's ID that you wish to stream.

1. List the cameras: 
   - `ffmpeg -list_devices true -f dshow -i dummy`

- Output may look something lik this:

```text
[dshow @ 000001d5b67ba0c0] "Full HD webcam" (video)
[dshow @ 000001d5b67ba0c0]   Alternative name "@device_pnp_\\?\usb#vid_0bda&pid_58b0&mi_00#7&35f82953&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
[dshow @ 000001d5b67ba0c0] "OBS Virtual Camera" (none)
[dshow @ 000001d5b67ba0c0]   Alternative name "@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
```

2. Transmit video using UDP from the target USB camera:
   - `ffmpeg -f dshow -i video="Full HD webcam" -framerate 24 -s 1280x720 -f mpegts udp://localhost:5000`

> **Note:** Replace `Full HD webcam` with the one you have identified in your system.

#### Questions

What is UDP and how can it be used to transmit video?

### 1.3 - Viewing the video streaming

- View the UDP stream:
  - `ffplay udp://localhost:5000`


### 1.4 - Capture frames from the stream

- Capture a frame from the stream every 5 seconds:
  - `ffmpeg -i udp://localhost:5000 -vf "fps=1/5" -f image2 output_%d.jpg`
