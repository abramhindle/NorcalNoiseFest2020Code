#!/bin/env python3
import os
import glob
import argparse
import random
import librosa as lib
import librosa.display
import numpy as np
import scipy
from tqdm import tqdm
import multiprocessing
from pippi import dsp
sr=48000
def prog_map(elms, f, desc="Synth", chunksize=1,procs=8,order=True):
    with tqdm(elms, desc=desc) as t:
        with multiprocessing.Pool(procs, initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
            if (order):
                pool = list(p.imap(f, t, chunksize=chunksize))
            else:
                pool = list(p.imap_unordered(f, t, chunksize=chunksize))
            return pool

import librosa

def mfcc(audio, nwin=256, nfft=512, fs=16000, nceps=13):
    #return librosa.feature.mfcc(y=audio, sr=44100, hop_length=nwin, n_mfcc=nceps)
    return [np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_fft=nfft, win_length=nwin,n_mfcc=nceps))]

def add_feature(mfcc1, rmsa1):
    tmfcc1 = np.zeros((mfcc1.shape[0],mfcc1.shape[1]+rmsa1.shape[0]))
    n = mfcc1.shape[0]
    m = mfcc1.shape[1]
    w = rmsa1.shape[0]
    tmfcc1[0:n,0:m] = mfcc1[0:n,0:m]
    tmfcc1[0:n,m:m+w]   = np.transpose(rmsa1[0:w,0:n])
    return tmfcc1

def std_mfcc(mfcc):
    return (mfcc - np.mean(mfcc, axis=0)) / np.std(mfcc, axis=0)

def calc_features(a1):
    fs = sr
    mfcc1 = mfcc(a1, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    mfcc1 = std_mfcc(mfcc1)
    rmsa1 = librosa.feature.rms(a1)
    cent1 = librosa.feature.spectral_centroid(y=a1, sr=fs)
    rolloff1 = librosa.feature.spectral_rolloff(y=a1, sr=fs, roll_percent=0.1)
    chroma_cq1 = librosa.feature.chroma_cqt(y=a1, sr=fs, n_chroma=10)
    onset_env1 = librosa.onset.onset_strength(y=a1, sr=sr)
    try:
        pulse1 = librosa.beat.plp(onset_envelope=onset_env1, sr=sr)
    except:
        pulse1 = np.ones((mfcc1.shape[0],))
    mfcc1 = add_feature(mfcc1, rmsa1)
    mfcc1 = add_feature(mfcc1, rolloff1/fs)
    mfcc1 = add_feature(mfcc1, cent1/fs)
    mfcc1 = add_feature(mfcc1, chroma_cq1)
    mfcc1 = add_feature(mfcc1, onset_env1.reshape(1,onset_env1.shape[0]))
    mfcc1 = add_feature(mfcc1, pulse1.reshape(1,onset_env1.shape[0]))
    return mfcc1

def cross_correlation(mfcc1, mfcc2, nframes):
    n1, mdim1 = mfcc1.shape
    n2, mdim2 = mfcc2.shape
    #print((nframes,(n1,mdim1),(n2,mdim2)))
    if (n2 <= nframes):
        t = np.zeros((nframes,mdim2))
        t[0:n2,0:mdim2] = mfcc2[0:n2,0:mdim2]
        mfcc2 = t
    if (n1 < nframes):
        t = np.zeros((nframes,mdim1))
        t[0:n1,0:mdim1] = mfcc2[0:n1,0:mdim1]
        mfcc1 = t
        n1 = nframes
    n = n1 - nframes + 1
    #c = np.zeros(min(n2,n))
    c = np.zeros(n)
    #for k in range(min(n2,n)):
    for k in range(n):
        cc = np.sum(np.multiply(mfcc1[k:k+nframes], mfcc2[:nframes]), axis=0)
        c[k] = np.linalg.norm(cc,1)
    return c



# cosine distance is A*B / sqrt(sumA^2)*sqrt(sumB^2)
def rosa_compare(a1, a2, fs=48000, trim=60*15, correl_nframes=50, plotit=True):
    sr = fs
    #print("Ref samples: %s Find samples: %s" % (a1.shape[0],a2.shape[0]))
    mfcc1 = calc_features(a1)
    mfcc2 = calc_features(a2)
    #mfcc1 = mfcc(a1, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    #mfcc2 = mfcc(a2, nwin=256, nfft=512, fs=fs, nceps=26)[0]
    c = cross_correlation(mfcc1, mfcc2, nframes=correl_nframes)
    max_k_index = np.argmax(c)
    # offset = max_k_index * (a1.shape[0]/rmsa1.shape[1]) / float(fs) # * over / sample rate
    #print("Best matching window: %s" % max_k_index)
    #print("mean %s std %s" % (np.mean(c) , np.std(c)))
    score = (c[max_k_index] - np.mean(c)) / (0.0000001 + np.std(c)) # standard score of peak
    return score

def parse_args():
    parser = argparse.ArgumentParser(description='Process some sounds.')
    parser.add_argument('-csv',default="out.csv",help="output csv file")
    parser.add_argument('sounds', metavar='N', type=str, nargs='+',
                    help='Sounds to load')
    args = parser.parse_args()
    return args

def mean_features(features):
    return np.mean(features, axis=0)

def calc_row_from_buffer(buffer):
    row = mean_features(calc_features(np.asarray(buffer[0:,0])))
    row = np.nan_to_num(row,nan=0.0)
    return row

def calc_row(filename):
    s = dsp.read(filename)
    return calc_row_from_buffer(s.frames)

def text_of_row(filename,row):
    return ",".join(['"'+filename+'"', ",".join([str(x) for x in row])])

def load_db(filename):
    with open(filename) as fd:
        lines = fd.readlines()
        # check for header?
        n = len(lines)
        w = len(lines[0].split(",")) - 1
        db = np.zeros((n,w))
        filenames  = []
        filenameid = {}
        i = 0
        for line in lines:
            s = line.strip().split(",")
            fname = s[0]
            fname = fname.strip('"')
            row = [float(x) for x in s[1:]]
            db[i,0:w] = row[0:w]
            filenames.append(fname)
            filenameid[fname] = i
            i+=1
        return {"rows":db,"filenames":filenames,"id":filenameid}

if __name__ == "__main__":
    args = parse_args()
    sounds = args.sounds
    with open(args.csv,"w") as fd:
        for sound in sounds:
            newsounds = [sound]
            if (os.path.isdir(sound)):
                newsounds = sorted(glob.glob(sound+"/*wav"))
            for sound in newsounds:
                row = calc_row(sound)
                out = text_of_row(sound, row)
                print(out)
                fd.write(out+"\n")

