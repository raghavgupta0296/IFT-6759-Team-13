import numpy as np
import tensorflow as tf
from utilities.sequencer import Sequencer



def preprocess_3h_sequence(batch):
    X = []
    y = []
    for sequence in batch:
        X.append([sequence[0]['image'], sequence[1]['image']]) # Image(T0-1), Image(T0)
        y.append([sequence[1]['GHI'], sequence[2]['GHI']])     # GHI(T0), GHI(T0 + 1)
    return np.array(X), np.array(y)


def preprocess_9h_sequence(batch):
    x = []
    y = []
    for sequence in batch:
        # Image(T0-2), Image(T0-1), Image(T0)
        x.append([sequence[0]['image'], sequence[2]['image'], sequence[4]['image']])
        # GHI(T0), GHI(T0 + 1), GHI(T0 + 3), GHI(T0 + 6)
        y.append([sequence[4]['GHI'], sequence[63]['GHI'], sequence[10]['GHI'], sequence[16]['GHI']])
    return np.array(x), np.array(y)


# Class used for loading data
class SequenceDataLoader_1h_int_3h_seq(tf.data.Dataset):
    def _generator(sequencer_object):
        while True:
            batch = sequencer_object.generate_batch()
            if not batch:
                break
            x, y = preprocess_3h_sequence(batch)
            yield x, y

    def __new__(cls, sequencer_object):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float64,
            output_shapes=((None, 2), (None, 2)),
            args=sequencer_object
        )


# Class used for loading data
class SequenceDataLoader_30min_int_9h_seq(tf.data.Dataset):
    def _generator(sequencer_object):
        while True:
            batch = sequencer_object.generate_batch()
            if not batch:
                break
            x, y = preprocess_3h_sequence(batch)
            yield x, y

    def __new__(cls, sequencer_object):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float64,
            output_shapes=((None, 3), (None, 4)),
            args=sequencer_object
        )


def benchmark(dataset, num_epochs=2):
    start_time = perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            sleep(0.05)
    tf.print("Execution time:", perf_counter() - start_time)