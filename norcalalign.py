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

def sim_tuples(sim, features, ttfeatures, t):
    (i,j) = t
    return (i,j,sim(features[j], ttfeatures[i]))

# Argh!
__db__ = None
__ttfeatures__ = None
__sim__ = None
__shared_arr__ = None
def get_db_ttfeatures():
    return (__db__,__ttfeatures)
def get_db_rows():
    return __db__["rows"]
def set_db(db):
    global __db__
    __db__ = db
def set_ttfeatures(ttfeatures):
    global __ttfeatures__
    __ttfeatures__ = ttfeatures
def get_ttfeatures():
    return __ttfeatures__
def get_sim():
    return __sim__
def set_sim(sim):
    global __sim__
    __sim__ = sim

def get_shared_arr():
    return __shared_arr__
def set_shared_arr(shared_arr):
    global __shared_arr__
    __shared_arr__ = shared_arr

def sim_tuples_global(t):
    return sim_tuples(get_sim(), get_db_rows(), get_ttfeatures(), t)

def process_row(i):
        features = get_db_rows()
        ttfeatures = get_ttfeatures()
        sim = get_sim()
        ttrow = ttfeatures[i]
        shared_arr = get_shared_arr()
        w = features.shape[0]
        # print(w,features.shape,features[0].shape,ttrow.shape)
        res = np.array([sim(feature, ttrow[0:]) for feature in features])
        buff = np.frombuffer(shared_arr.get_obj())
        # print(w,len(res),buff.shape,res.shape)
        buff[i*w:(i+1)*w] = res[0:w]
        return True

def calculate_similarity_old(ttfeatures, db, sim=cosine_sim):
    set_db(db)
    set_ttfeatures(ttfeatures)
    set_sim(sim)
    ttmax = ttfeatures.shape[0]
    size = ttmax
    w = len(db["rows"])
    zerossize = ttmax*len(db["rows"])
    shared_arr = multiprocessing.Array(ctypes.c_double, zerossize)
    set_shared_arr(shared_arr)
    ttres = np.zeros((ttmax,len(db["rows"])))
    # args = itertools.product(range(ttmax),range(len(db["rows"])))
    args = range(ttmax)
    #for t in prog_map(args, process_row, chunksize=1, order=False, total=size):
    #    (i,v) = t
    #    ttres[i,0:] = v[0:]
    ## row by row
    prog_map(args, process_row, chunksize=1, order=False, total=size)
    for i in range(ttmax):
        ttres[i,0:] = shared_arr[i*w:(i+1)*w]
    return ttres

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
