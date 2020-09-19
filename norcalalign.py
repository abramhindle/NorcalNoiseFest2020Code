#!/bin/env python3
import argparse
from pippi.oscs import Osc, Osc2d, Pulsar, Pulsar2d, Alias, Bar
from pippi import dsp, interpolation, wavetables, fx, oscs,soundpipe
from pippi.soundbuffer import SoundBuffer
from pippi.wavesets import Waveset
from pippi import dsp, fx
from tqdm import tqdm
import multiprocessing
import random
import librosa as lib
import librosa.display
import numpy as np
from pippi import dsp, noise
import scipy
from helpers import *
import param_generation as pg
import norcalextract
import functools
import itertools
import ctypes
import scipy.spatial
import scipy
sr=48000

def prog_map(elms, f, desc="Synth", chunksize=1,procs=8,order=True,total=None):
    with tqdm(elms, desc=desc, total=total) as t:
        with multiprocessing.Pool(procs, initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
            if (order):
                pool = list(p.imap(f, t, chunksize=chunksize))
            else:
                pool = list(p.imap_unordered(f, t, chunksize=chunksize))
            return pool


@functools.lru_cache(maxsize=256)
def get_samples(filename):
    return dsp.read(filename)

def rmse_distance(mfcc1,mfcc2):
    n = mfcc1.shape[0]
    return 1.0 - np.sqrt(np.sum((mfcc1 - mfcc2)**2)/n)

def cosine_sim(a,b):
    n = a.shape[0]
    aab  = np.average(a * b )
    asqa = np.average(np.square(a))
    asqb = np.average(np.square(b))
    return aab/np.sqrt(asqa*asqb)

def calculate_similarity(ttfeatures, db, sim='cosine'):
    return scipy.spatial.distance.cdist(ttfeatures, db["rows"], metric=sim)
    

def fit_to_input(ttres):
    ttfilt = np.nan_to_num(ttres,nan=-np.Inf)    
    ttfit = np.argmax(ttfilter,axis=1)
    return ttfit

def render(ttfeatures, ttfit, db,freq=1.0):
    out = dsp.buffer(length=512*ttfeatures.shape[0]/sr)
    skip = sr//int((1/freq)*512)
    steps = ttargmin.shape[0] // skip
    for i in range(steps):
        j = i * skip
        index = int(ttargmin[j])
        s = get_samples(db["filenames"][index])
        t = j * 512 / sr
        print(t,index)
        out.dub(s.buff,t)
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Align a long sound to a DB of sounds')
    parser.add_argument('-i', help='Input wave')
    parser.add_argument('-out',default='output.wav', help='the rendered output')
    parser.add_argument('-csv',default="out.csv",help="input csv file")
    parser.add_argument('-s', default=1.0,help='frequency per second of sounds')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    filename = args.i
    print("Loading %s" % filename)
    input_snd = dsp.read(args.i)
    freq = float(args.s)
    print("Loading %s" % args.csv)
    db = norcalextract.load_db(args.csv)
    print("Feature Extracting %s" % filename)
    ttfeatures = norcalextract.calc_features(np.asarray(input_snd.frames[0:,0]))
    print(ttfeatures.shape)
    print(db["rows"].shape)
    print("Calculating Similarity")
    ttres = calculate_similarity(ttfeatures, db)
    print("Filtering")
    ttfit = fit_to_input(ttres)
    print(ttfit.shape)
    print("Rendering")
    out = render(ttfeatures, ttfit, db, freq)
    out.write(args.out)
