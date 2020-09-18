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
sr=48000

def prog_map(elms, f, desc="Synth", chunksize=1,procs=4,order=True):
    with tqdm(elms, desc=desc) as t:
        with multiprocessing.Pool(procs, initializer=tqdm.set_lock,
              initargs=(tqdm.get_lock(),)) as p:
            if (order):
                pool = list(p.imap(f, t, chunksize=chunksize))
            else:
                pool = list(p.imap_unordered(f, t, chunksize=chunksize))
            return pool

def parse_args():
    parser = argparse.ArgumentParser(description='Generate some sounds')
    parser.add_argument('-prefix', default='synth', help='wav prefix')
    parser.add_argument('-out',default='output', help='outputdir')
    parser.add_argument('-csv',default="out.csv",help="output csv file")
    parser.add_argument('-n', default=1000,help='How many sounds')
    parser.add_argument('-t', default=1.0,help='sample length')
    args = parser.parse_args()
    return args

def generate_and_save(p,filename):
    s=pg.Synth(p)
    s.buff.write(filename)
    return (filename,norcalextract.calc_row_from_buffer(s.buff.frames))


def generate_and_save_tuple(tup):
    (p,filename) = tup
    return generate_and_save(p,filename)

def gen_filename(spec, x):
    return spec["out"]+"/"+spec["prefix"]+'{:06d}'.format(x)+".wav"
def generate(csvfd,spec):
    pg.min_length = spec["length"]
    filenames = [(pg.RandomParams(), gen_filename(spec,x))
                 for x in range(spec["n"])]
    for tup in prog_map(filenames, generate_and_save_tuple, chunksize=16, order=False):
        (filename,row) = tup
        csvfd.write(norcalextract.text_of_row(filename,row))
        



if __name__ == "__main__":
    args = parse_args()
    spec = {"prefix":"synth", "out":"output", "n":1000,"t":1.0}
    spec["prefix"] = args.prefix
    spec["out"] = args.out
    spec["n"] = int(args.n)
    spec["length"] = float(args.t)
    with open(args.csv,"w") as csvfd:
        generate(csvfd,spec)
