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
import scipy.io.wavfile
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

def parse_args():
    parser = argparse.ArgumentParser(description='split some sounds')
    parser.add_argument('-i',  help='input sound')
    parser.add_argument('-out',default='output', help='outputdir')
    parser.add_argument('-csv',default="out.csv",help="output csv file")
    parser.add_argument('-prefix',default="split",help="Prefix")
    parser.add_argument('-t', default=1.0,help='sample length (seconds)')
    parser.add_argument('-sr',default=sr,help='assumed sample rate, try to use 48000')
    args = parser.parse_args()
    return args

__sound__ = None
def set_sound(sound):
    global __sound__
    __sound__ = sound

def get_sound(sound):
    return __sound__

def get_clip( sample, samplelen):
    o = np.zeros((samplelen,1))
    o[0:samplelen,0] =  __sound__[sample:(sample+samplelen),0]
    return o

def generate_and_save(i,sample,samplelen,filename):
    clip = get_clip( sample, samplelen)
    # p.write(clip, filename
    scipy.io.wavfile.write(filename, sr, (32000*clip).astype('int16'))
    return (filename,norcalextract.calc_row_from_buffer(clip))
    
def generate_and_save_tuple(tup):
    return generate_and_save(*tup)

def gen_filename(spec, x, format='{:06d}'):
    return spec["out"]+"/"+spec["prefix"]+format.format(x)+".wav"

def generate(csvfd,spec):
    pg.min_length = spec["length"]
    sound = dsp.read(spec["filename"])
    # dumb multiprocessing hack
    set_sound(sound.frames)
    n = int(sr * spec["length"])
    nsamps = sound.frames.shape[0]
    steps = nsamps // n
    filenames = [(x,x*n,n,gen_filename(spec,x)) for x in range(steps)]
    for tup in prog_map(filenames, generate_and_save_tuple, chunksize=16, order=False):
        (filename,row) = tup
        csvfd.write(norcalextract.text_of_row(filename,row)+"\n")
        



if __name__ == "__main__":
    args = parse_args()
    spec = {}
    spec["filename"] = args.i
    spec["prefix"] = args.prefix
    spec["out"] = args.out
    spec["csv"] = args.csv
    spec["length"] = float(args.t)
    sr = int(args.sr)
    with open(args.csv,"w") as csvfd:
        generate(csvfd,spec)
