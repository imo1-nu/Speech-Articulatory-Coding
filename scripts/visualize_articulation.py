"""Render saved EMA using reference-style vocal tract outlines and matching audio.

The projected outlines are traced from the official lock-rock demo. Motion comes
from the supplied EMA, through an illustrative display rig. This is not the
authors' original renderer, nor a reconstruction of the speaker's anatomy.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import RBFInterpolator
import soundfile as sf


FPS = 50
REFERENCE = Path(__file__).parent / "assets" / "vocal_tract_reference.json"
# Pixels in the reference projection. EMA itself remains unchanged.
CENTERS = np.array([[650., 505.], [480., 475.], [300., 545.],
                    [293., 655.], [222., 500.], [232., 610.]])
# Order: TD, TB, TT, LI, UL, LL. Screen Y increases downward.
GAINS = np.array([[-12., -34.], [-12., -34.], [-12., -38.],
                  [-8., -20.], [-7., -9.], [-10., -20.]])


def skin_weights(points):
    """Local, continuous jaw/lip deformations; the cranial outline stays fixed."""
    weights = np.zeros((len(points), 6))
    for channel, spread in ((3, [90., 100.]), (4, [65., 60.]), (5, [70., 65.])):
        distance = (points - CENTERS[channel]) / spread
        weights[:, channel] = np.exp(-.5 * (distance**2).sum(axis=1))
    chin = (points - [420., 815.]) / [190., 90.]
    weights[:, 3] += .4 * np.exp(-.5 * (chin**2).sum(axis=1))
    weights /= np.maximum(1., weights.sum(axis=1, keepdims=True))
    return weights


def prepare_layers(reference):
    """Precompute a small deformation rig for the reference's vector contours."""
    tongue_nodes = np.array([CENTERS[2], CENTERS[1], CENTERS[0],
                             [420., 780.], [680., 780.], [730., 700.]])
    tongue_drivers = np.zeros((6, 6))
    tongue_drivers[0, 2] = tongue_drivers[1, 1] = tongue_drivers[2, 0] = 1
    tongue_drivers[3, 3] = .25
    tongue_drivers[4, 3] = .15
    tongue_rig = RBFInterpolator(tongue_nodes, tongue_drivers,
                                 kernel="thin_plate_spline", degree=1)
    layers = []
    for layer in reference["layers"]:
        for polygon in layer["polygons"]:
            points = np.asarray(polygon, dtype=float)
            role = layer["role"]
            if role == "tongue":
                weights = tongue_rig(points)
            elif role in ("head", "head_highlight", "floor", "teeth"):
                weights = skin_weights(points)
            elif role == "palate":
                weights = skin_weights(points)
                weights[:, :4] = 0
                weights[:, 5] = 0
            else:
                weights = np.zeros((len(points), 6))
            layers.append((points, weights, tuple(layer["color"])))
    return layers


def render_frame(reference, layers, ema_frame):
    """Deform vector geometry, then supersample its flat-shaded projection."""
    width, height = reference["canvas"]
    image = Image.new("RGB", (width*2, height*2), tuple(reference["background"]))
    draw = ImageDraw.Draw(image)
    displacement = np.asarray(ema_frame).reshape(6, 2) * GAINS
    for points, weights, color in layers:
        vertices = (points + weights @ displacement) * 2
        draw.polygon([tuple(point) for point in vertices], fill=color)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="NPZ saved by resynthesize.sh --features")
    parser.add_argument("audio", type=Path, help="Corresponding audio file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output MP4")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".mp4":
        parser.error("Output must have an .mp4 extension")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with np.load(args.features, allow_pickle=False) as features:
        ema = features["ema"]
    if ema.ndim != 2 or ema.shape[1] != 12 or not len(ema) or not np.isfinite(ema).all():
        parser.error("Expected finite EMA data with shape (frames, 12)")
    audio = sf.info(args.audio)
    if abs(len(ema)/FPS - audio.duration) > .1:
        parser.error("Audio and EMA durations differ by more than 0.1 seconds")
    frames = min(len(ema), round(audio.duration*FPS))
    reference = json.loads(REFERENCE.read_text())
    layers = prepare_layers(reference)
    width, height = reference["canvas"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sparc-animation-") as temporary:
        encoded = Path(temporary) / "video.mp4"
        comment = ("EMA-driven illustrative deformation; not measured anatomy or the original "
                   f"authors' renderer. Reference: {reference['source']}")
        command = ["ffmpeg", "-v", "error", "-nostdin", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                   "-framerate", str(FPS), "-i", "pipe:0", "-i", str(args.audio),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                   "-metadata", "title=SPARC vocal tract visualization",
                   "-metadata", f"comment={comment}", str(encoded)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for frame in range(frames):
                    encoder.stdin.write(render_frame(reference, layers, ema[frame]).tobytes())
                    if frame % 250 == 0:
                        print(f"Rendered {frame+1}/{frames} frames", flush=True)
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("FFmpeg could not encode the animation")
        shutil.move(str(encoded), args.output)
    print(f"Video: {args.output} ({frames/FPS:.2f} s, {FPS} fps, {width}x{height})")


if __name__ == "__main__":
    main()
