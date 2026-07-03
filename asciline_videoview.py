#!/usr/bin/env python3
"""
ASCILINE Core Logic - Unified Terminal and Frontend Implementation
================================================================

Extracts the core logic from the ASCILINE project to provide:
1. Fixed-screen terminal output (using ANSI escape codes)
2. Web-friendly frame generation (for frontend consumption)

Dependencies: opencv-python, numpy, and pyav (optional, recommended for AV1)
"""

import argparse
import sys
import time
import json
import html
import numpy as np
import os
import shutil

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import av
except ImportError:
    av = None


class VideoDecoder:
    """
    Unified Video Decoder that extracts frame pairs (gray, bgr) from a video file.
    Tries PyAV first (essential for AV1 playback on platforms without hardware-accelerated
    OpenCV AV1 support), and falls back to OpenCV.
    """
    def __init__(self, path: str, cols: int, rows: int, skip_gray: bool = False):
        self.path = path
        self.cols = cols
        self.rows = rows
        self._size = (cols, rows)
        self.skip_gray = skip_gray
        self.use_av = False
        self.container = None
        self.cap = None

        # Attempt to initialize PyAV
        if av is not None:
            try:
                self.container = av.open(path)
                # Select the first video stream
                self.stream = next(s for s in self.container.streams if s.type == "video")
                self.vid_w = self.stream.codec_context.width
                self.vid_h = self.stream.codec_context.height
                self.fps = float(self.stream.average_rate) if self.stream.average_rate else 24.0
                self._generator = self.container.decode(video=0)
                self.use_av = True
            except Exception:
                if self.container:
                    self.container.close()
                    self.container = None

        # Fall back to OpenCV if PyAV failed or isn't installed
        if not self.use_av:
            if cv2 is None:
                raise ImportError(
                    "Neither 'av' (PyAV) nor 'cv2' (OpenCV) is available. Please install at least one."
                )
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise FileNotFoundError(f"Could not open video file: {path!r}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
            self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        if self.use_av:
            try:
                frame = next(self._generator)
                bgr = frame.to_ndarray(format="bgr24")
                
                # Resize using OpenCV (fast) or numpy fallback
                if cv2 is not None:
                    small = cv2.resize(bgr, self._size, interpolation=cv2.INTER_LINEAR)
                    gray = None if self.skip_gray else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                else:
                    # Pure numpy nearest-neighbor resizing fallback
                    h, w = bgr.shape[:2]
                    y_indices = np.linspace(0, h - 1, self._size[1], dtype=int)
                    x_indices = np.linspace(0, w - 1, self._size[0], dtype=int)
                    small = bgr[np.ix_(y_indices, x_indices)]
                    gray = None if self.skip_gray else np.dot(small[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
                
                return gray, small
            except (StopIteration, Exception):
                raise StopIteration
        else:
            # OpenCV backend
            ok, frame = self.cap.read()
            if not ok:
                raise StopIteration
            small = cv2.resize(frame, self._size, interpolation=cv2.INTER_LINEAR)
            gray = None if self.skip_gray else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return gray, small

    def release(self):
        if self.container:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class AsciiMapper:
    """
    Converts Gray + BGR frame pairs into colored ASCII strings.
    Supports terminal ANSI true-color output (with RLE) and web HTML formats.
    """
    DEFAULT_PALETTE = list(
        " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    )

    def __init__(self, palette: list[str] | None = None, quantize_bits: int = 0, color_mode: int = 4):
        p = palette or self.DEFAULT_PALETTE
        self._n = len(p)
        self._lut = np.array(p, dtype="U1")
        self._qb = quantize_bits  # Right shift bit amount for color quantization
        self.color_mode = color_mode

    def convert_terminal(self, gray: np.ndarray, bgr: np.ndarray) -> str:
        """
        Convert frame to ANSI-colored terminal ASCII with Run-Length Encoding (RLE).
        """
        if gray is None and self.color_mode != 1 and self._n > 1:
            gray = np.dot(bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

        H, W = bgr.shape[:2]

        # Step 1: Grayscale to character mapping
        if self._n > 1 and gray is not None:
            gray_32 = gray.astype(np.uint32)
            indices = (gray_32 * self._n) >> 8
            np.clip(indices, 0, self._n - 1, out=indices)
            char_matrix = self._lut[indices]
        else:
            # If palette has only 1 char (pixel mode)
            char_matrix = np.full((H, W), self._lut[0], dtype="U1")

        # Mode 1: Pure black and white text output (no colors)
        if self.color_mode == 1:
            lines = ["".join(row) for row in char_matrix]
            return "\n".join(lines)

        # Step 2: Color quantization and BGR -> RGB view conversion
        rgb = bgr[:, :, ::-1]
        if self._qb > 0:
            rgb = (rgb >> self._qb) << self._qb

        # Step 3: Construction of colored string using RLE
        lines = []
        prev_r = prev_g = prev_b = -1

        for row_idx in range(H):
            row_chars = char_matrix[row_idx]
            row_colors = rgb[row_idx]
            buf = []

            for col_idx in range(W):
                r, g, b = int(row_colors[col_idx, 0]), \
                          int(row_colors[col_idx, 1]), \
                          int(row_colors[col_idx, 2])

                if r != prev_r or g != prev_g or b != prev_b:
                    buf.append(f"\033[38;2;{r};{g};{b}m")
                    prev_r, prev_g, prev_b = r, g, b

                buf.append(row_chars[col_idx])

            lines.append("".join(buf))

        return "\033[0m" + "\n".join(lines) + "\033[0m"

    def convert_web(self, gray: np.ndarray, bgr: np.ndarray) -> str:
        """
        Convert frame to HTML representation with inline style span coloring and RLE.
        """
        if gray is None and self.color_mode != 1 and self._n > 1:
            gray = np.dot(bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

        H, W = bgr.shape[:2]

        if self._n > 1 and gray is not None:
            gray_32 = gray.astype(np.uint32)
            indices = (gray_32 * self._n) >> 8
            np.clip(indices, 0, self._n - 1, out=indices)
            char_matrix = self._lut[indices]
        else:
            char_matrix = np.full((H, W), self._lut[0], dtype="U1")

        # Mode 1: Pure black and white output (no HTML colors)
        if self.color_mode == 1:
            lines = []
            for row_idx in range(H):
                row_chars = char_matrix[row_idx]
                buf = [html.escape(ch) for ch in row_chars]
                lines.append("".join(buf))
            return "\n".join(lines)

        rgb = bgr[:, :, ::-1]
        if self._qb > 0:
            rgb = (rgb >> self._qb) << self._qb

        lines = []
        for row_idx in range(H):
            row_chars = char_matrix[row_idx]
            row_colors = rgb[row_idx]
            buf = []
            prev_r = prev_g = prev_b = -1
            span_open = False

            for col_idx in range(W):
                r, g, b = int(row_colors[col_idx, 0]), \
                          int(row_colors[col_idx, 1]), \
                          int(row_colors[col_idx, 2])
                char = html.escape(row_chars[col_idx])

                if r != prev_r or g != prev_g or b != prev_b:
                    if span_open:
                        buf.append("</span>")
                    buf.append(f'<span style="color:rgb({r},{g},{b})">')
                    span_open = True
                    prev_r, prev_g, prev_b = r, g, b

                buf.append(char)

            if span_open:
                buf.append("</span>")
            lines.append("".join(buf))

        return "\n".join(lines)


class AsciiRenderer:
    """
    High-level renderer executing the conversion pipeline:
    VideoDecoder -> AsciiMapper -> Playback / JSON Export.
    """
    CHAR_RATIO = 0.45  # Character aspect ratio adjustment for terminal cells

    def __init__(
        self,
        path: str,
        cols: int = 80,
        rows: int = 40,
        palette: list[str] | None = None,
        quantize_bits: int = 0,
        color_mode: int = 4,
        pixel_mode: bool = False,
    ):
        self.path = path
        self.cols = cols
        self.rows = rows
        self.palette = palette
        self.quantize_bits = quantize_bits
        self.color_mode = color_mode
        self.pixel_mode = pixel_mode

        # If pixel mode is enabled, we force the palette to contain only solid block character
        if self.pixel_mode and not self.palette:
            self.palette = ["█"]

        # Probe video dimensions and FPS
        probe = VideoDecoder(path, 2, 2)
        self.fps = probe.fps
        self.vid_w = probe.vid_w
        self.vid_h = probe.vid_h
        probe.release()

        # Handle auto-fit size for terminal
        self._auto_fit = (self.cols == 0 or self.rows == 0)

    def _determine_terminal_dims(self) -> tuple[int, int]:
        """Calculate aspect-ratio preserving dimensions fitting within terminal size."""
        term = shutil.get_terminal_size(fallback=(80, 40))
        t_cols = term.columns
        t_lines = term.lines - 2  # Safety padding margin

        aspect = self.vid_h / self.vid_w
        
        # Determine maximum safe limits
        safe_cols = min(t_cols, 160)
        
        if self.vid_w >= self.vid_h:  # Landscape
            cols = safe_cols
            rows = max(1, int(cols * aspect * self.CHAR_RATIO))
            if rows > t_lines:
                rows = t_lines
                cols = max(1, int(rows / (aspect * self.CHAR_RATIO)))
        else:  # Portrait
            rows = t_lines
            cols = max(1, int(rows / (aspect * self.CHAR_RATIO)))
            if cols > safe_cols:
                cols = safe_cols
                rows = max(1, int(cols * aspect * self.CHAR_RATIO))

        return cols, rows

    def frames_terminal(self, cols: int, rows: int):
        decoder = VideoDecoder(self.path, cols, rows, skip_gray=self.pixel_mode)
        mapper = AsciiMapper(self.palette, self.quantize_bits, self.color_mode)
        try:
            for gray, bgr in decoder:
                yield mapper.convert_terminal(gray, bgr)
        finally:
            decoder.release()

    def frames_web(self, cols: int, rows: int):
        decoder = VideoDecoder(self.path, cols, rows, skip_gray=self.pixel_mode)
        mapper = AsciiMapper(self.palette, self.quantize_bits, self.color_mode)
        try:
            for gray, bgr in decoder:
                yield mapper.convert_web(gray, bgr)
        finally:
            decoder.release()

    def play_fixed_screen(self, delay: float | None = None):
        """
        Play video in a fixed terminal screen with correct aspect-ratio preserving auto-fit and padding.
        """
        cols = self.cols
        rows = self.rows
        if self._auto_fit:
            cols, rows = self._determine_terminal_dims()

        frame_delay = delay if delay is not None else 1.0 / self.fps
        stdout = sys.stdout

        # Precalculate centering padding
        term = shutil.get_terminal_size(fallback=(80, 40))
        t_cols = term.columns
        t_lines = term.lines - 2

        pad_y = max(0, (t_lines - rows) // 2)
        pad_x_str = " " * max(0, (t_cols - cols) // 2)

        try:
            # Enter alternate buffer, hide cursor, clear screen
            stdout.write("\033[?1049h\033[?25l\033[2J")
            stdout.flush()

            for frame in self.frames_terminal(cols, rows):
                t0 = time.perf_counter()
                
                # Apply centering padding
                if pad_x_str:
                    frame = pad_x_str + frame.replace('\n', '\n' + pad_x_str)
                if pad_y > 0:
                    frame = ('\n' * pad_y) + frame

                # Move cursor to home and write frame
                stdout.write("\033[H" + frame)
                stdout.flush()

                # Pace the playback
                elapsed = time.perf_counter() - t0
                wait = frame_delay - elapsed
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            pass
        finally:
            # Reset normal terminal buffer
            stdout.write("\033[0m\033[?25h\033[?1049l\n")
            stdout.flush()

    def generate_web_data(self) -> dict:
        """
        Generate JSON-serializable dictionary with HTML frames and metadata.
        """
        cols = self.cols if self.cols > 0 else 80
        rows = self.rows if self.rows > 0 else 40
        frames = list(self.frames_web(cols, rows))

        return {
            "type": "ascii-video",
            "format": "html",
            "fps": self.fps,
            "cols": cols,
            "rows": rows,
            "frame_count": len(frames),
            "frames": frames,
            "pixel_mode": self.pixel_mode,
        }


def build_parser():
    p = argparse.ArgumentParser(
        description="ASCII Video Converter - Core logic for Terminal and Web rendering."
    )
    p.add_argument("video", help="Path to video file")
    p.add_argument(
        "-c", "--cols", type=int, default=80, help="ASCII grid width in columns (use 0 for terminal auto-fit)"
    )
    p.add_argument(
        "-r", "--rows", type=int, default=40, help="ASCII grid height in rows (use 0 for terminal auto-fit)"
    )
    p.add_argument(
        "-p",
        "--palette",
        type=str,
        default=None,
        help="Custom character palette as a continuous string",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Override frame rate delay in seconds",
    )
    p.add_argument(
        "--out",
        "--output",
        dest="out",
        choices=["terminal", "web"],
        default="terminal",
        help="Output mode: 'terminal' (real-time playback) or 'web' (dumps JSON output)",
    )
    p.add_argument(
        "--pixel",
        "--PIXEL",
        "--px",
        "--PX",
        action="store_true",
        dest="pixel",
        help="Enable pixel-perfect block rendering mode using solid blocks",
    )
    p.add_argument(
        "--color",
        "--COLOR",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=4,
        dest="color",
        help="Color rendering mode: 1=B&W, 2=512 colors, 3=32K, 4=262K, 5=16M (default: 4)",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    palette = list(args.palette) if args.palette else None

    # Handle smart defaults: if pixel mode is on, default to Mode 5 (16M colors)
    # unless the user explicitly specified a color mode in the command line
    has_color_arg = any(arg in sys.argv for arg in ["--color", "--COLOR"])
    color_mode = args.color if has_color_arg else (5 if args.pixel else 4)

    # Map color mode choices to quality bit shift quantization:
    # 5 (16M Color) -> 0 shift
    # 4 (262K Color) -> 2 shift
    # 3 (32K Color) -> 3 shift
    # 2 (512 Color) -> 5 shift
    # 1 (B&W) -> 0 shift
    qb = {5: 0, 4: 2, 3: 3, 2: 5, 1: 0}.get(color_mode, 2)

    renderer = AsciiRenderer(
        path=args.video,
        cols=args.cols,
        rows=args.rows,
        palette=palette,
        quantize_bits=qb,
        color_mode=color_mode,
        pixel_mode=args.pixel,
    )

    if args.out == "terminal":
        renderer.play_fixed_screen(delay=args.delay)
    elif args.out == "web":
        data = renderer.generate_web_data()
        json.dump(data, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

# ==============================================================================
# HOW TO USE & ARGUMENTS EXPLANATION
# ==============================================================================
#
# USAGE EXAMPLES:
# --------------
# 0. Play video as pixel style, toString
#    python3 logic.py video.mp4  -c 220 -r 90  --pixel
#
# 1. Play video inside terminal (fixed-screen, zero-flicker, auto-fitting):
#    python3 logic.py input.mp4 --out terminal -c 0 -r 0
#
# 2. Play pixel-perfect colored block video in terminal:
#    python3 logic.py input.mp4 --out terminal --pixel -c 0 -r 0
#
# 3. Export ASCII animation as JSON with highest quality 16M color mode:
#    python3 logic.py input.mp4 --out web --color 5 -c 100 -r 45 > output.json
#
# ARGUMENTS EXPLANATION:
# ---------------------
# positional arguments:
#   video                Path to input video file (e.g. MP4, AVI, WebM).
#
# optional arguments:
#   -h, --help           Show this help message and exit.
#   -c COLS, --cols COLS
#                        ASCII grid width in columns (default: 80).
#                        Set to 0 to enable terminal width auto-fit.
#   -r ROWS, --rows ROWS
#                        ASCII grid height in rows (default: 40).
#                        Set to 0 to enable terminal height auto-fit.
#   -p PALETTE, --palette PALETTE
#                        Custom palette of characters (default: 93-char preset).
#                        Must be passed as a continuous string, ordered from darkest
#                        to brightest intensities.
#   --delay DELAY        Override framerate pacing delay in seconds (default: video FPS).
#   --out {terminal,web}, --output {terminal,web}
#                        Output visualization mode (default: terminal).
#                        'terminal': Real-time playback inside the terminal buffer.
#                        'web': Dumps a clean JSON structure containing all HTML frame strings.
#   --pixel, --PIXEL, --px, --PX
#                        Enable pixel-perfect block rendering. Replaces standard
#                        ASCII character mapping with solid block characters (█),
#                        making the output visually close to a normal video.
#   --color, --COLOR {1,2,3,4,5}
#                        Color rendering quality modes:
#                        - 1: Black and White (No colors, text only).
#                        - 2: 512 Colors (9-bit quantized color space).
#                        - 3: 32K Colors (15-bit quantized color space).
#                        - 4: 262K Colors (18-bit quantized color space, default).
#                        - 5: 16M Colors (Full 24-bit True Color).
#
# ==============================================================================
# WEB VISUALIZATION LOGIC (JS/TS IMPLEMENTATION)
# ==============================================================================
# Save the following HTML file to play the exported JSON files directly in the browser:
#
# ```html
# <!DOCTYPE html>
# <html lang="en">
# <head>
#   <meta charset="UTF-8">
#   <title>ASCILINE Web Viewer</title>
#   <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;600&display=swap" rel="stylesheet">
#   <style>
#     body {
#       background: radial-gradient(circle at center, #111322 0%, #07080f 100%);
#       color: #e2e8f0;
#       font-family: 'Outfit', sans-serif;
#       margin: 0;
#       display: flex;
#       flex-direction: column;
#       align-items: center;
#       justify-content: center;
#       min-height: 100vh;
#       overflow: hidden;
#     }
#     h1 {
#       font-weight: 300;
#       letter-spacing: 2px;
#       margin-bottom: 20px;
#       text-shadow: 0 0 10px rgba(99, 102, 241, 0.4);
#     }
#     #player-wrapper {
#       background: rgba(21, 23, 38, 0.7);
#       backdrop-filter: blur(10px);
#       border: 1px solid rgba(255, 255, 255, 0.1);
#       border-radius: 16px;
#       padding: 24px;
#       box-shadow: 0 20px 50px rgba(0,0,0,0.6);
#       display: flex;
#       flex-direction: column;
#       align-items: center;
#       transition: all 0.3s ease;
#     }
#     #screen {
#       font-family: "Courier New", Courier, monospace;
#       line-height: 1;
#       letter-spacing: 0;
#       white-space: pre;
#       background-color: #030305;
#       color: #fff;
#       padding: 16px;
#       border-radius: 8px;
#       border: 1px solid rgba(255,255,255,0.05);
#       margin-bottom: 20px;
#       overflow: auto;
#       font-size: 8px;
#       font-weight: bold;
#       box-shadow: inset 0 0 20px rgba(0,0,0,0.8);
#     }
#     .controls {
#       display: flex;
#       align-items: center;
#       gap: 16px;
#       width: 100%;
#     }
#     button {
#       background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
#       border: none;
#       color: white;
#       padding: 10px 20px;
#       border-radius: 8px;
#       cursor: pointer;
#       font-weight: 600;
#       font-family: inherit;
#       box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
#       transition: all 0.2s ease;
#     }
#     button:hover {
#       transform: translateY(-2px);
#       box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
#     }
#     button:active {
#       transform: translateY(0);
#     }
#     input[type="range"] {
#       flex-grow: 1;
#       accent-color: #6366f1;
#       height: 6px;
#       border-radius: 3px;
#       background: #2d314d;
#       outline: none;
#     }
#     .time-info {
#       font-size: 14px;
#       color: #94a3b8;
#       min-width: 100px;
#       text-align: right;
#       font-variant-numeric: tabular-nums;
#     }
#     .upload-area {
#       border: 2px dashed rgba(99, 102, 241, 0.5);
#       padding: 30px;
#       border-radius: 12px;
#       text-align: center;
#       cursor: pointer;
#       transition: background 0.3s;
#       background: rgba(99, 102, 241, 0.05);
#     }
#     .upload-area:hover {
#       background: rgba(99, 102, 241, 0.1);
#       border-color: #6366f1;
#     }
#   </style>
# </head>
# <body>
#
#   <h1>ASCILINE Dynamic Web Player</h1>
#
#   <div id="upload-container" class="upload-area">
#     <p>Drag and drop your ASCII video JSON file here, or click to browse</p>
#     <input type="file" id="file-selector" accept=".json" style="display:none;">
#   </div>
#
#   <div id="player-wrapper" style="display: none;">
#     <pre id="screen"></pre>
#     <div class="controls">
#       <button id="btn-play">Play</button>
#       <input type="range" id="scrubber" min="0" value="0">
#       <span id="btn-time" class="time-info">00:00 / 00:00</span>
#     </div>
#   </div>
#
#   <script>
#     let animation = null;
#     let frameIdx = 0;
#     let running = false;
#     let ticker = null;
#
#     const dropZone = document.getElementById('upload-container');
#     const fileInput = document.getElementById('file-selector');
#     const player = document.getElementById('player-wrapper');
#     const screen = document.getElementById('screen');
#     const playBtn = document.getElementById('btn-play');
#     const scrubber = document.getElementById('scrubber');
#     const timeDisplay = document.getElementById('btn-time');
#
#     dropZone.addEventListener('click', () => fileInput.click());
#     dropZone.addEventListener('dragover', (e) => { e.preventDefault(); });
#     dropZone.addEventListener('drop', (e) => {
#       e.preventDefault();
#       const file = e.dataTransfer.files[0];
#       if (file) handleFile(file);
#     });
#     fileInput.addEventListener('change', (e) => {
#       const file = e.target.files[0];
#       if (file) handleFile(file);
#     });
#
#     function handleFile(file) {
#       const reader = new FileReader();
#       reader.onload = (evt) => {
#         try {
#           animation = JSON.parse(evt.target.result);
#           if (animation.type !== 'ascii-video') {
#             alert('Invalid JSON structure. Needs type="ascii-video".');
#             return;
#           }
#           dropZone.style.display = 'none';
#           player.style.display = 'flex';
#           initializePlayback();
#         } catch (err) {
#           alert('Error parsing JSON file.');
#         }
#       };
#       reader.readAsText(file);
#     }
#
#     function formatTime(frame, fps) {
#       const totalSecs = frame / fps;
#       const m = Math.floor(totalSecs / 60).toString().padStart(2, '0');
#       const s = Math.floor(totalSecs % 60).toString().padStart(2, '0');
#       return `${m}:${s}`;
#     }
#
#     function initializePlayback() {
#       frameIdx = 0;
#       running = false;
#       if (ticker) clearInterval(ticker);
#       playBtn.textContent = 'Play';
#       scrubber.max = animation.frame_count - 1;
#       scrubber.value = 0;
#       drawCurrentFrame();
#     }
#
#     function drawCurrentFrame() {
#       if (!animation) return;
#       screen.innerHTML = animation.frames[frameIdx];
#       scrubber.value = frameIdx;
#       timeDisplay.textContent = `${formatTime(frameIdx, animation.fps)} / ${formatTime(animation.frame_count, animation.fps)}`;
#     }
#
#     playBtn.addEventListener('click', () => {
#       if (running) {
#         pause();
#       } else {
#         play();
#       }
#     });
#
#     scrubber.addEventListener('input', () => {
#       frameIdx = parseInt(scrubber.value);
#       drawCurrentFrame();
#     });
#
#     function play() {
#       running = true;
#       playBtn.textContent = 'Pause';
#       ticker = setInterval(() => {
#         frameIdx = (frameIdx + 1) % animation.frame_count;
#         drawCurrentFrame();
#       }, 1000 / animation.fps);
#     }
#
#     function pause() {
#       running = false;
#       playBtn.textContent = 'Play';
#       clearInterval(ticker);
#     }
#   </script>
# </body>
# </html>
# ```