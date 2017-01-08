from glob import glob
import json
import os.path
import random

from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.models import Sequential, model_from_json
import numpy as np


import logging

logger = None
def init_logging():
    global logger

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
init_logging()


def get_char_indices(data):
    chars = sorted(list(set(data)))
    char_to_index = {
        c: i for i, c in enumerate(chars)
    }
    index_to_char = {
        i: c for i, c in enumerate(chars)
    }
    return char_to_index, index_to_char


def get_train_data(data, seq_length, step=1):
    char_to_index, index_to_char = get_char_indices(data)
    n_chars = len(char_to_index)

    seqs = []
    next_char = []
    for offset in xrange(0, len(data) - seq_length, step):
        seqs.append(data[offset:offset + seq_length])
        next_char.append(data[offset + seq_length])
    n_samples = len(seqs)

    X = np.zeros((n_samples, seq_length, n_chars), dtype=np.bool)
    y = np.zeros((n_samples, n_chars), dtype=np.bool)

    for i, seq in enumerate(seqs):
        for char_index, char in enumerate(seq):
            X[i, char_index, char_to_index[char]] = 1
        y[i, char_to_index[next_char[i]]] = 1

    return X, y


class GenerationCallback(Callback):
    SAMPLES_DIR = 'samples'
    SAMPLE_PATTERN = os.path.join(SAMPLES_DIR, 'samples-{epoch:02d}-{diversity:.1f}.txt')
    DIVERSITIES = (0.2, 0.5, 0.7, 1.0, )

    def __init__(self, wrapper, output_dir, sample_length=None, sample_pattern=None):
        self._wrapper = wrapper
        self.sample_pattern = self.SAMPLE_PATTERN
        self.output_dir = output_dir

        if sample_pattern:
            self.sample_pattern = sample_pattern

        self.sample_length = self._wrapper._input_length * 20
        if sample_length is not None:
            self.sample_length = sample_length
        super(GenerationCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        seed = self._wrapper.get_random_seed()

        for diversity in self.DIVERSITIES:
            sample = self._wrapper.generate(
                seed=seed,
                length=self.sample_length,
                diversity=diversity,
            ).encode('utf-8')

            if self.output_dir:
                self.store_sample(sample, epoch, diversity)
            else:
                logger.debug('Sample %d-%.1f "%s"', epoch, diversity, sample)

    def store_sample(self, sample, epoch, diversity):
        dirname = os.path.join(self.output_dir, self.SAMPLES_DIR, )
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        filename = os.path.join(self.output_dir, self.sample_pattern.format(
            epoch=epoch,
            diversity=diversity,
        ))
        with open(filename, 'w') as f:
            logger.debug('Storing samples in %s', filename)
            f.write(sample)


class RNNWrapper(object):
    MODEL_FILENAME = 'model.json'
    WRAPPER_FILENAME = 'wrapper.json'

    def __init__(self, data, output_dim, input_length, layers=1, dropout=None, output_dir=None, sample_length=None, initial_epoch=0):
        self._data = data
        self._output_dim = output_dim
        self._input_length = input_length
        self._layers = layers
        self._dropout = dropout
        self._output_dir = output_dir
        self._sample_length = sample_length
        self._initial_epoch = initial_epoch

        self._input_dim = len(self.char_to_index)
        logger.debug('Input dim %d', self._input_dim)

        self._model = self._create(output_dim, self._input_dim, input_length, layers, dropout)

        self._callbacks = []
        self._callbacks.append(GenerationCallback(self, output_dir=output_dir, sample_length=sample_length))
        if output_dir is not None:
            self._callbacks.append(self.get_save_callback(output_dir))
            self.store_model()
        self._fit = False

    def store_model(self):
        path = os.path.join(self._output_dir, self.MODEL_FILENAME)
        logger.info('Storing model in %s', path)
        with open(path, 'w') as f:
            f.write(self._model.to_json())

        path = os.path.join(self._output_dir, self.WRAPPER_FILENAME)
        logger.info('Storing wrapper in %s', path)
        with open(path, 'w') as f:
            f.write(json.dumps({
                'output_dim': self._output_dim,
                'input_length': self._input_length,
                'layers': self._layers,
                'dropout': self._dropout,
                'output_dir': self._output_dir,
            }))
        try:
            from keras.utils.visualize_util import plot
            path = os.path.join(self._output_dir, self.MODEL_IMAGE_FILENAME)
            logger.info('Storing model image in %s', path)
            plot(self._model,
                 to_file=path, show_shapes=True
            )
        except ImportError:
            logger.info('Can\'t plot model', exc_info=True)
            pass

    WEIGHTS_DIR = 'weights'
    WEIGHTS_FILENAME_PATTERN = os.path.join(WEIGHTS_DIR, 'weights-{epoch:02d}-{loss:.4f}.hdf5')
    WEIGHTS_FILENAME_GLOB = os.path.join(WEIGHTS_DIR, 'weights-*-*.hdf5')

    @classmethod
    def get_save_callback(cls, dirname, model_name=None):
        weigths_dirname = os.path.join(dirname, cls.WEIGHTS_DIR)
        if not os.path.exists(weigths_dirname):
            os.mkdir(weigths_dirname)

        pattern = str(cls.WEIGHTS_FILENAME_PATTERN)
        if model_name is not None:
            pattern = model_name + '-' + pattern
        callback = ModelCheckpoint(
            os.path.join(dirname, pattern),
            monitor='loss', verbose=1, save_best_only=True, mode='min',
        )
        return callback

    @classmethod
    def get_best_model(cls, data, from_dir):
        model_filename = os.path.join(from_dir, cls.MODEL_FILENAME)
        wrapper_filename = os.path.join(from_dir, cls.WRAPPER_FILENAME)
        weights_filenames = glob(os.path.join(from_dir, cls.WEIGHTS_FILENAME_GLOB))
        assert os.path.exists(model_filename), \
            '%s must exist to load from dir' % cls.MODEL_FILENAME
        assert os.path.exists(wrapper_filename), \
            '%s must exist to load from dir' % cls.WRAPPER_FILENAME

        best_epoch = None
        best_loss = None
        best_weights = None

        for weights_filename in weights_filenames:
            _, epoch, loss = weights_filename[:-5].split('-')
            epoch = int(epoch)
            loss = float(loss)
            if best_epoch is None or best_loss > loss:
                best_epoch = epoch
                best_loss = loss
                best_weights = weights_filename
        if best_epoch is not None:
            logger.debug('Best epoch %d with loss %.4f at file %s', best_epoch + 1, best_loss, best_weights)
        else:
            logger.debug('No weights files were found')

        with open(wrapper_filename, 'r') as f:
            wrapper_params = json.load(f)

        if best_epoch is not None:
            # in keras logs epochs are 1-based
            # and in callbacks epochs are 0-based
            # so for epoch N in logs, filename is epoch - 1
            initial_epoch = best_epoch + 1
        else:
            initial_epoch = None

        wrapper = cls(data=data, initial_epoch=initial_epoch, **wrapper_params)
        if best_weights is not None:
            wrapper._model.load_weights(best_weights)
        return wrapper

    def _create(self, output_dim, input_dim, input_length, layers=1, dropout=None, ):
        model = Sequential()
        for i in xrange(layers):
            if dropout is not None and dropout > 0:
                model.add(Dropout(dropout))
            if i == 0:
                # input layer
                model.add(LSTM(output_dim, input_dim=input_dim, input_length=input_length, ))
        model.add(Dense(input_dim,))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def fit(self, nb_epoch=0, ):
        self._fit = True
        if self._initial_epoch > 1:
            nb_epoch = nb_epoch + self._initial_epoch
        X, y = get_train_data(self._data, self._input_length, )
        self._model.fit(X, y, batch_size=128, nb_epoch=nb_epoch, callbacks=self._callbacks, initial_epoch=self._initial_epoch)

    def sample(self, preds, diversity=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def get_random_seed(self, length=None):
        if length is None:
            length = self._input_length
        index = random.randint(0, len(self._data) - length)
        return self._data[index:index + length]

    def generate(self, seed=None, length=1, diversity=1.0):
        assert self._fit

        if seed is None:
            seed = self.get_random_seed()
        initial_seed = unicode(seed)
        result = []
        for _ in xrange(length):
            seq = np.zeros((1, self._input_length, self._input_dim))
            for i, char in enumerate(seed):
                seq[0, i, self.char_to_index[char]] = 1

            preds = self._model.predict(seq, )[0]
            new_char_index = self.sample(preds, diversity)
            new_char = self.index_to_char[new_char_index]
            result.append(new_char)
            seed += new_char
            seed = seed[1:]
        return initial_seed + ''.join(result)
