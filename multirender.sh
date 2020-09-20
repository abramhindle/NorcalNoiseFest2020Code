#!/bin/bash
CSVS="tt-mistake.csv all.csv 8bit.csv 10d.csv recordings.csv gpu.csv combined.csv tt.csv mz412.csv"
S="0.1 0.2 0.3"
D="cosine sqeuclidean"
MVIS=`ls MVI_4???.wav`
for csv in $CSVS; do
	for s in $S; do
		for d in $D; do
			for mv in $MVIS; do
				echo python3 norcalalign.py -i $mv -o $mv.$csv.$s.$d.wav -csv $csv -s $s -sim $d
				echo python3 norcalringmod.py -i $mv.$csv.$s.$d.wav -m $mv -o $mv.$csv.$s.$d.wav.ring.wav -ws $s
				#python3 norcalalign.py -i $mv -o $mv.$csv.$s.$d.wav -csv $csv -s $s -sim $d
			done
		done
	done
done
