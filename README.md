# QuickClip

One video in, vertical captioned clips out.

Give it a YouTube link or a video file. It finds the loudest stretches of the video and
renders each one as a 1080x1920 clip: the full picture over a blurred background, with
word-timed captions, at -14 LUFS.

## Install

1. Python 3.10 or newer.
2. ffmpeg on PATH.
3. `pip install -r requirements.txt`

## Use the page

```
python quickclip.py
```

Your browser opens at http://127.0.0.1:8000. Paste a link, press **Make clips**, and the
clips show up with a player and a download link. The page is only reachable from this PC.

## Use the command line

```
python quickclip.py https://www.youtube.com/watch?v=VIDEO_ID
python quickclip.py my_video.mp4 --clips 5 --seconds 20
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--clips` | how many clips | 3 |
| `--seconds` | length of each clip | 30 |
| `--port` | port of the page | 8000 |

## Good to know

- Clips land in `clips/`, downloaded videos in `downloads/`.
- The first run downloads the speech model (about 500 MB).
- Captions run on the CPU, about 20 seconds per clip.
- `python quickclip.py selftest` checks the clip picker.
