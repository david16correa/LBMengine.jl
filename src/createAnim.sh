#!/bin/bash
ffmpeg -loglevel quiet -framerate 30 -i tmp/%d.png -c:v libx264 -pix_fmt yuv420p anims/output.mp4
rm -r tmp
