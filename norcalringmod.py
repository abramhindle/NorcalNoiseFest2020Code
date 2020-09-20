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
import scipy.signal
sr=48000


def parse_args():
    parser = argparse.ArgumentParser(description='modulate 1 sound by the other')
    parser.add_argument('-i',  help='input sound to module')
    parser.add_argument('-m',  help='module sound')
    parser.add_argument('-ws', default=0.1, help='window size in seconds')
    parser.add_argument('-out',default='output.wav', help='Output wav file')
    parser.add_argument('-sr',default=sr,help='assumed sample rate, try to use 48000')
    parser.add_argument('-n',default=True, help='normalize')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    print("Loading")
    input_snd = dsp.read(args.i)
    mod_snd   = dsp.read(args.m)
    outfile = args.out
    wss = float(args.ws)
    sr = int(args.sr)
    ws = int(wss * sr)
    window = np.hanning(ws)/ws
    inputframes = input_snd.frames[0:,0]
    print("Convolution")
    env = scipy.signal.fftconvolve(np.abs(mod_snd.frames[0:,0]), window, mode='valid')
    if (args.n):
        # normalize
        m = np.max(env)
        env /= m
    print(mod_snd.frames[0:,0].shape, env.shape, inputframes.shape)
    l = max(inputframes.shape[0],env.shape[0])
    outputsnd = np.zeros((l,))
    outputsnd[0:env.shape[0]] += env
    outputsnd[0:inputframes.shape[0]] *= inputframes
    scipy.io.wavfile.write(outfile, sr, (32000*outputsnd).astype('int16'))
