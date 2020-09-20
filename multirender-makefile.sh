#!/bin/bash
CSVS="tt-mistake.csv all.csv 8bit.csv 10d.csv recordings.csv gpu.csv combined.csv tt.csv mz412.csv"
S="0.1 0.2 0.3"
D="cosine sqeuclidean"
MVIS=`ls MVI_4???.wav`
T=`echo -e "\t"`
X=""
V=""
for csv in $CSVS; do
	for s in $S; do
		for d in $D; do
			for mv in $MVIS; do
				W1=$mv.$csv.$s.$d.wav
				W2=$mv.$csv.$s.$d.wav.ring.wav
				W3=videos/$mv.$csv.$s.$d.wav.ring.mkv
				W4=videos/$mv.$csv.$s.$d.wav.mkv
				echo ${W1} :
				echo -e \\tpython3 norcalalign.py -i $mv -o ${W1} -csv $csv -s $s -sim $d
				echo ${W2} : ${W1}
				echo -e \\tpython3 norcalringmod.py -i ${W1} -m $mv -o ${W2} -ws $s
				echo ${W3} : ${W2}
				echo -e \\tbash mkvid.sh $mv ${W2} ${W3}
				echo ${W4} : ${W1}
				echo -e \\tbash mkvid.sh $mv ${W1} ${W4}
				X="$X $W1 $W2"
				V="$V $W3 $W4"
			done
		done
	done
done
echo "audio: $X"
echo "video: $V"
