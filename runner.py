import argparse
import os.path


def is_dir(dirname):
    """Checks if a path is an actual directory"""
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', type=int, help='Number of neurons on LSTM layer.', )
parser.add_argument('--length', '-l', type=int, help='Length of training sequences', )
parser.add_argument('--epoch', '-e', type=int, help='Number of epochs to train', default=1)
parser.add_argument('--layers', type=int, help='Number of LSTM layers.', default=1)
parser.add_argument('--dropout', '-d', type=float, help='Dropout argument', default=0)
parser.add_argument('--out', '-o', type=is_dir)
parser.add_argument('--file', '-f', type=argparse.FileType('r'), help='data file', required=True)
parser.add_argument('--resume', '-r', action='store_true', help='Resume training from dir', )
parser.add_argument('--sample', '-s', type=int, help='Sample length', )
parser.add_argument('-seed', type=int, help='numpy initial seed')

args = parser.parse_args()
if args.seed:
    import numpy as np
    np.random.seed(args.seed)

from rnn import RNNWrapper

data = args.file.read().decode('utf-8').lower().replace('\n', '')

if not args.resume:
    wrapper = RNNWrapper(data,
        output_dim=args.num,
        input_length=args.length,
        layers=args.layers,
        dropout=args.dropout,
        output_dir=args.out,
        sample_length=args.sample,
    )
else:
    wrapper = RNNWrapper.get_best_model(data, args.out)
wrapper.fit(nb_epoch=args.epoch)
