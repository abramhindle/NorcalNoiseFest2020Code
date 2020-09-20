#!/bin/bash
F1=`basename -s .MOV $1`
F2=`basename -s .wav $F1`
MOVNAME=${F2}.MOV
MKVNAME=$3
ffmpeg -i $MOVNAME -i $2 -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 $MKVNAME
