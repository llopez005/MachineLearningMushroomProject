import pandas as pd
import tensorflow as tf

TRAIN_PATH = "./training_data.csv"
TEST_PATH = "./testing_data.csv"

CSV_COLUMN_NAMES = ['pois','cap.shape', 'cap.surface',
                    'cap.color', 'bruises', 'odor',
                    'gill.attachment', 'gill.spacing',
                    'gill.size', 'gill.color', 'stalk.shape',
                    'stalk.surface.above.ring',
                    'stalk.surface.below.ring', 'stalk.color.above.ring',
                    'stalk.color.below.ring',
                    'veil.color', 'ring.number', 'ring.type',
                    'spore.print.color', 'population', 'habitat']
CLASS = ['Edible', 'Poisonous']

def load_data(y_name='pois'):
    train = pd.read_csv(TRAIN_PATH, header=0)
    train['pois'].replace('e', 0, inplace=True)
    train['pois'].replace('p', 1, inplace=True)
    train = pd.get_dummies(train, columns=CSV_COLUMN_NAMES[1:])
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(TEST_PATH, header=0)
    test['pois'].replace('e', 0, inplace=True)
    test['pois'].replace('p', 1, inplace=True)
    test = pd.get_dummies(test, columns=CSV_COLUMN_NAMES[1:])
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(800).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None or zero"
    dataset = dataset.batch(batch_size)
    return dataset
