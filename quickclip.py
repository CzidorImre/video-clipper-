"""QuickClip: one video in, a few vertical captioned clips out.

    python quickclip.py                           open the page in your browser
    python quickclip.py <link or file>            make clips from the command line
    python quickclip.py <link or file> --clips 5 --seconds 20

It cuts the loudest stretches of the video. Needs ffmpeg on PATH and:
    pip install -r requirements.txt
"""

import argparse
import contextlib
import functools
import json
import re
import subprocess
import sys
import threading
import wave
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote

HERE = Path(__file__).resolve().parent
OUT = HERE / "clips"
DOWNLOADS = HERE / "downloads"

# The whole 16:9 picture, centred over a blurred copy of itself, 1080x1920.
FIT = (
    "[0:v]split[a][b];"
    "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
    "boxblur=40:6,eq=brightness=-0.10[bg];"
    "[b]scale=1080:1920:force_original_aspect_ratio=decrease,setsar=1[fg];"
    "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[f]"
)
STYLE = "Fontname=Arial,Fontsize=11,Bold=1,Outline=1,Alignment=2,MarginV=64"


# ---------------------------------------------------------------- engine


def ffmpeg(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-nostats", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd,
    )
    if done.returncode:
        raise RuntimeError(done.stderr[-600:])
    return done.stderr


def download(url: str) -> Path:
    import yt_dlp

    opts = {
        "format": "bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(DOWNLOADS / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url)
        return Path(ydl.prepare_filename(info)).with_suffix(".mp4")


def loudness(src: Path) -> list[float]:
    """Mean loudness (LUFS) of every second of the video."""
    log = ffmpeg("-i", str(src), "-vn", "-af", "ebur128", "-f", "null", "-")
    seconds: dict[int, list[float]] = {}
    for t, m in re.findall(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*(-?[\d.]+)", log):
        seconds.setdefault(int(float(t)), []).append(max(float(m), -70.0))
    return [sum(v) / len(v) for _, v in sorted(seconds.items())]


def pick(loud: list[float], n: int, size: int) -> list[int]:
    """Start seconds of the n loudest windows that do not overlap."""
    size = min(size, len(loud))
    score = {s: sum(loud[s : s + size]) for s in range(len(loud) - size + 1)}
    starts: list[int] = []
    for s in sorted(score, key=lambda s: score[s], reverse=True):
        if len(starts) == n:
            break
        if all(abs(s - t) >= size for t in starts):
            starts.append(s)
    return sorted(starts)


def stamp(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


@functools.lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel

    # CPU int8 runs on any PC; use device="cuda" on a box with a working CUDA setup.
    return WhisperModel("small", device="cpu", compute_type="int8")


def write_captions(wav: Path, srt: Path) -> bool:
    """Three words at a time. False when nobody speaks."""
    import numpy as np  # comes with faster-whisper

    # Read the wav ourselves: faster-whisper's own file decoding breaks on some PyAV versions.
    with wave.open(str(wav)) as f:
        audio = np.frombuffer(f.readframes(-1), np.int16).astype(np.float32) / 32768
    segments, _ = whisper().transcribe(audio, word_timestamps=True)
    words = [w for seg in segments for w in seg.words or []]
    blocks = [words[i : i + 3] for i in range(0, len(words), 3)]
    srt.write_text(
        "".join(
            f"{i}\n{stamp(b[0].start)} --> {stamp(b[-1].end)}\n"
            f"{''.join(w.word for w in b).strip()}\n\n"
            for i, b in enumerate(blocks, 1)
        ),
        encoding="utf-8",
    )
    return bool(blocks)


def render(src: Path, start: int, seconds: int) -> Path:
    # A plain name: it goes into an ffmpeg filter and onto the page.
    name = f"{re.sub(r'[^A-Za-z0-9_-]', '_', src.stem)}_{start}s"
    wav, srt = OUT / f"{name}.wav", OUT / f"{name}.srt"
    cut = ("-ss", str(start), "-t", str(seconds), "-i", str(src.resolve()))
    ffmpeg(*cut, "-vn", "-ac", "1", "-ar", "16000", str(wav))
    spoken = write_captions(wav, srt)
    # ffmpeg runs inside OUT so the subtitles filter gets a plain file name (no C: to escape)
    text = f"subtitles={srt.name}:force_style='{STYLE}'" if spoken else "null"
    ffmpeg(
        *cut, "-filter_complex", f"{FIT};[f]{text}[v]", "-map", "[v]", "-map", "0:a",
        "-af", "loudnorm=I=-14", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", f"{name}.mp4", cwd=OUT,
    )
    wav.unlink()
    srt.unlink()
    return OUT / f"{name}.mp4"


def make_clips(
    source: str, count: int = 3, seconds: int = 30, say: Callable[[str], None] = print
) -> list[Path]:
    OUT.mkdir(exist_ok=True)
    source = source.strip().strip('"')
    if source.startswith("http"):
        say("Downloading the video...")
        src = download(source)
    else:
        src = Path(source)
        if not src.is_file():
            raise RuntimeError(f"File not found: {source}")
    say("Listening for the loudest moments...")
    loud = loudness(src)
    if not loud:
        raise RuntimeError("This video has no sound.")
    starts = pick(loud, count, seconds)
    say("Loading the speech model (the first run downloads about 500 MB)...")
    whisper()
    clips = []
    for i, start in enumerate(starts, 1):
        say(f"Making clip {i} of {len(starts)}...")
        clips.append(render(src, start, seconds))
    say(f"Done: {len(clips)} clips in {OUT}")
    return clips


# ---------------------------------------------------------------- page

STATE: dict[str, object] = {"busy": False, "message": "Paste a link, then press Make clips."}
LOCK = threading.Lock()

HTML = b"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>QuickClip</title>
<style>
:root{color-scheme:dark;--bg:#0f1115;--card:#181b22;--line:#2a2f3a;--text:#e8eaf0;
 --dim:#8d94a5;--go:#f5c518}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{margin:0;font-size:26px;letter-spacing:-.02em} h1 b{color:var(--go)}
.sub{margin:4px 0 24px;color:var(--dim)}
form{display:grid;grid-template-columns:1fr 96px 96px auto;gap:12px;align-items:end;
 background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
label{display:grid;gap:4px;font-size:12px;color:var(--dim)}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid var(--line);
 background:var(--bg);color:var(--text);font:inherit}
input:focus-visible,button:focus-visible{outline:2px solid var(--go);outline-offset:2px}
button{padding:10px 18px;border:0;border-radius:8px;background:var(--go);color:#111;
 font:600 15px system-ui,sans-serif;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
#msg{margin:16px 2px 24px;color:var(--dim);min-height:1.5em;overflow-wrap:anywhere}
#clips{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;
 overflow:hidden}
video{display:block;width:100%;aspect-ratio:9/16;background:#000}
figcaption{display:flex;justify-content:space-between;gap:8px;padding:10px 12px;font-size:12px;
 color:var(--dim);overflow-wrap:anywhere}
a{color:var(--go);text-decoration:none;white-space:nowrap} a:hover{text-decoration:underline}
@media (max-width:640px){form{grid-template-columns:1fr 1fr} .wide,button{grid-column:1/-1}}
</style></head><body><main>
<h1>Quick<b>Clip</b></h1>
<p class="sub">One video in, vertical captioned clips out.</p>
<form id="f">
 <label class="wide">YouTube link or video file path
  <input name="source" required placeholder="https://www.youtube.com/watch?v=..."></label>
 <label>Clips<input name="clips" type="number" min="1" max="10" value="3"></label>
 <label>Seconds<input name="seconds" type="number" min="5" max="120" value="30"></label>
 <button id="go">Make clips</button>
</form>
<p id="msg" role="status"></p>
<div id="clips"></div>
</main><script>
const $ = id => document.getElementById(id);
let shown = null;
async function tick() {
  const s = await (await fetch("/status")).json();
  $("msg").textContent = s.message;
  $("go").disabled = s.busy;
  const key = s.clips.join("|");
  if (key === shown) return;
  shown = key;
  $("clips").innerHTML = s.clips.map(n => {
    const url = "/clips/" + encodeURIComponent(n);
    return `<figure><video controls preload="metadata" src="${url}"></video>` +
      `<figcaption><span>${n}</span><a download href="${url}">Download</a></figcaption></figure>`;
  }).join("");
}
$("f").onsubmit = async e => {
  e.preventDefault();
  await fetch("/make", {method: "POST", body: new URLSearchParams(new FormData($("f")))});
  tick();
};
setInterval(tick, 1500);
tick();
</script></body></html>"""


def job(source: str, count: int, seconds: int) -> None:
    try:
        make_clips(source, count, seconds, say=lambda m: STATE.update(message=m))
    except Exception as e:  # anything that goes wrong is shown on the page
        STATE["message"] = f"Failed: {str(e)[-600:]}"
    finally:
        STATE["busy"] = False


class Page(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def reply(self, body: bytes, kind: str = "text/plain", code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self.reply(HTML, "text/html; charset=utf-8")
        elif self.path == "/status":
            found = sorted(OUT.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            # A clip that is still being written is not listed yet.
            clips = [p.name for p in found if not p.with_suffix(".srt").exists()]
            self.reply(json.dumps({**STATE, "clips": clips}).encode(), "application/json")
        elif self.path.startswith("/clips/"):
            self.clip(unquote(self.path[len("/clips/") :]))
        else:
            self.reply(b"not found", code=404)

    def clip(self, name: str) -> None:
        path = OUT / Path(name).name  # only a file directly inside clips/
        if path.suffix != ".mp4" or not path.is_file():
            return self.reply(b"not found", code=404)
        size = path.stat().st_size
        first, last = 0, size - 1
        # Browsers ask for byte ranges to seek inside a video.
        ranged = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
        if ranged:
            first = int(ranged[1])
            last = min(int(ranged[2]), last) if ranged[2] else last
            if first > last:
                return self.reply(b"bad range", code=416)
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(first)
            with contextlib.suppress(ConnectionError):  # the browser stopped loading
                self.wfile.write(f.read(last - first + 1))

    def do_POST(self) -> None:
        # Only this page may start a job, not a page from another site open in the browser.
        origin = self.headers.get("Origin")
        if self.path != "/make" or (origin and origin != f"http://{self.headers.get('Host')}"):
            return self.reply(b"not allowed", code=403)
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode())
        try:
            source = form["source"][0]
            count = min(max(int(form.get("clips", ["3"])[0]), 1), 10)
            seconds = min(max(int(form.get("seconds", ["30"])[0]), 5), 120)
        except (KeyError, ValueError):
            return self.reply(b"bad form", code=400)
        with LOCK:
            if STATE["busy"]:
                return self.reply(b"busy", code=409)
            STATE.update(busy=True, message="Starting...")
        threading.Thread(target=job, args=(source, count, seconds), daemon=True).start()
        self.reply(b"", code=202)


def serve(port: int = 8000, open_browser: bool = True) -> None:
    OUT.mkdir(exist_ok=True)
    # On Windows the default (reuse on) would silently share a port another program is using.
    ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), Page)  # this PC only
    except OSError:
        sys.exit(f"Port {port} is in use. Try: python quickclip.py --port {port + 1}")
    url = f"http://127.0.0.1:{port}"
    print(f"QuickClip is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


# ---------------------------------------------------------------- command line


def selftest() -> None:
    quiet = [-40.0] * 200
    loud = quiet[:50] + [-10.0] * 30 + quiet[80:150] + [-15.0] * 30 + quiet[180:]
    assert pick(loud, 2, 30) == [50, 150], pick(loud, 2, 30)
    assert pick(loud, 1, 30) == [50]
    assert pick([-20.0] * 10, 3, 30) == [0]  # shorter than one clip
    assert stamp(3725.5) == "01:02:05,500"
    print("ok")


def main() -> None:
    p = argparse.ArgumentParser(description="One video in, vertical captioned clips out.")
    p.add_argument("source", nargs="?", help="YouTube link or video file; leave out for the page")
    p.add_argument("--clips", type=int, default=3, help="how many clips (default 3)")
    p.add_argument("--seconds", type=int, default=30, help="length of each clip (default 30)")
    p.add_argument("--port", type=int, default=8000, help="port of the page (default 8000)")
    a = p.parse_args()
    if a.source is None:
        serve(a.port)
    elif a.source == "selftest":
        selftest()
    else:
        try:
            make_clips(a.source, a.clips, a.seconds)
        except Exception as e:
            sys.exit(f"Failed: {e}")


if __name__ == "__main__":
    main()
