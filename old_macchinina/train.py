#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]
    [--transfer=<transfer>]

Options:
    -h --help              Show this screen.
    --transfer=<transfer>  Path to transfer learning model.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    # comment = args['--comment']
    transfer = args['--transfer']
    if transfer is not None:
        train(cfg, tubs, model, model_type, transfer = transfer)
    else:
        train(cfg, tubs, model, model_type)


if __name__ == "__main__":
    main()
