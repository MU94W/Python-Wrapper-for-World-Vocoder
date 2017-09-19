from __future__ import division, print_function

import os
from shutil import rmtree
import argparse

import numpy as np

import matplotlib      # Remove this line if you don't need them
matplotlib.use('Agg')  # Remove this line if you don't need them
import matplotlib.pyplot as plt

import soundfile as sf
# import librosa
import pyworld as pw


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--frame_period", type=float, default=5.0)
parser.add_argument("-s", "--speed", type=int, default=1)


EPSILON = 1e-8

def savefig(filename, figlist, log=True):
    #h = 10
    n = len(figlist)
    # peek into instances
    f = figlist[0]
    if len(f.shape) == 1:
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i+1)
            if len(f.shape) == 1:
                plt.plot(f)
                plt.xlim([0, len(f)])
    elif len(f.shape) == 2:
        Nsmp, dim = figlist[0].shape
        #figsize=(h * float(Nsmp) / dim, len(figlist) * h)
        #plt.figure(figsize=figsize)
        plt.figure()
        for i, f in enumerate(figlist):
            plt.subplot(n, 1, i+1)
            if log:
                x = np.log(f + EPSILON)
            else:
                x = f + EPSILON
            plt.imshow(x.T, origin='lower', interpolation='none', aspect='auto', extent=(0, x.shape[0], 0, x.shape[1]))
    else:
        raise ValueError('Input dimension must < 3.')
    plt.savefig(filename)
    # plt.close()


def main(args):
    if os.path.isdir('test'):
        rmtree('test')
    os.mkdir('test')

    #x, fs = sf.read('utterance/vaiueo2d.wav')
    x, fs = sf.read('utterance/p226_002.wav')
    # x, fs = librosa.load('utterance/vaiueo2d.wav', dtype=np.float64)

    # 1. A convient way
    f0, sp, ap = pw.wav2world(x, fs)    # use default options
    y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

    # 2. Step by step
    # 2-1 Without F0 refinement
    _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
                    channels_in_octave=2,
                    frame_period=args.frame_period,
                    speed=args.speed)
    _sp = pw.cheaptrick(x, _f0, t, fs)
    _ap = pw.d4c(x, _f0, t, fs)
    _y = pw.synthesize(_f0, _sp, _ap, fs, args.frame_period)
    # librosa.output.write_wav('test/y_without_f0_refinement.wav', _y, fs)
    sf.write('test/y_without_f0_refinement.wav', _y, fs)

    # 2-2 DIO with F0 refinement (using Stonemask)
    f0 = pw.stonemask(x, _f0, t, fs)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    y = pw.synthesize(f0, sp, ap, fs, args.frame_period)
    # librosa.output.write_wav('test/y_with_f0_refinement.wav', y, fs)
    sf.write('test/y_with_f0_refinement.wav', y, fs)

    # 2-3 Harvest with F0 refinement (using Stonemask)
    _f0_h, t_h = pw.harvest(x, fs)
    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
    ap_h = pw.d4c(x, f0_h, t_h, fs)
    y_h = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
    # librosa.output.write_wav('test/y_harvest_with_f0_refinement.wav', y_h, fs)
    sf.write('test/y_harvest_with_f0_refinement.wav', y_h, fs)

    # 2-4 DIO with F0 refinement (using Stonemask). Code and restore sp, ap.
    code_sp = pw.code_spectral_envelope(sp, fs, 80)
    code_ap = pw.code_aperiodicity(ap, fs)
    fft_size = (sp.shape[1] - 1) * 2
    rest_sp = pw.decode_spectral_envelope(code_sp, fs, fft_size)
    rest_ap = pw.decode_aperiodicity(code_ap, fs, fft_size)
    y_r = pw.synthesize(f0, rest_sp, rest_ap, fs, args.frame_period)
    sf.write('test/y_with_f0_refinement_code_and_restore.wav', y_r, fs)
    print("fft size: {:d}".format(fft_size))
    print("coded sp shape: ({:d}, {:d})".format(code_sp.shape[0], code_sp.shape[1]))
    print("coded ap shape: ({:d}, {:d})".format(code_ap.shape[0], code_ap.shape[1]))

    # 2-5 DIO with F0 refinement (using Stonemask). Code and restore sp, ap. frame_shift: 12.5 ms, frame_length: 50.0 ms
    f0_xx, t_xx = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
                         channels_in_octave=2,
                         frame_period=12.5,
                         speed=args.speed)
    f0_xx = pw.stonemask(x, f0_xx, t_xx, fs)
    sp_xx = pw.cheaptrick(x, f0_xx, t_xx, fs)
    ap_xx = pw.d4c(x, f0_xx, t_xx, fs)
    code_sp_xx = pw.code_spectral_envelope(sp_xx, fs, 80)
    code_ap_xx = pw.code_aperiodicity(ap_xx, fs)
    fft_size = (sp_xx.shape[1] - 1) * 2
    rest_sp_xx = pw.decode_spectral_envelope(code_sp_xx, fs, fft_size)
    rest_ap_xx = pw.decode_aperiodicity(code_ap_xx, fs, fft_size)
    y_r_xx = pw.synthesize(f0_xx, rest_sp_xx, rest_ap_xx, fs, 12.5)
    sf.write('test/y_with_f0_refinement_code_and_restore_frame_period_12.5.wav', y_r_xx, fs)
    print("coded sp_xx shape: ({:d}, {:d})".format(code_sp_xx.shape[0], code_sp_xx.shape[1]))
    print("coded ap_xx shape: ({:d}, {:d})".format(code_ap_xx.shape[0], code_ap_xx.shape[1]))


    # Comparison
    savefig('test/wavform.png', [x, _y, y, y_h, y_r, y_r_xx])
    savefig('test/sp.png', [_sp, sp, sp_h, rest_sp, rest_sp_xx])
    savefig('test/ap.png', [_ap, ap, ap_h, rest_ap, rest_ap_xx], log=False)
    savefig('test/f0.png', [_f0, f0, f0_h, f0_xx])

    print('Please check "test" directory for output files')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
