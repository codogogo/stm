import argparse
import os
import tensorflow as tf
import numpy as np
import pickle
from helpers import io_helper
from helpers import data_shaper
from embeddings import text_embeddings
from models import wordpair_classifier
from ml import trainer
from ml import loss_functions

def build_feed_dict_func(model, data, config = None, predict = False):
	x_pairs, y = zip(*data)
	x1s = [x[0] for x in x_pairs]
	x2s = [x[1] for x in x_pairs]
	drp = 0.5

	fd = model.get_feed_dict(x1s, x2s, y, 1.0 if predict else drp, not predict)
	return fd, y


parser = argparse.ArgumentParser(description='Trains a model for classifying lexico-semantic relations.')
parser.add_argument('data', help='A path to the file containing training examples. Each training example should have the format: "first_word TAB second_word TAB label"')
parser.add_argument('embs', help='A path to the file containing pre-trained (unspecialized, distributional) word embeddings')
parser.add_argument('output', help='A file path to which to store (serialize, pickle) the model.')
parser.add_argument('--slice', type=int, help='Number (K) of tensor slices (default = 5)')
parser.add_argument('-s', '--specsize', type=int, help='The size (n) of the specialized word vectors (default 100)')
parser.add_argument('-l', '--learningrate', type=float, help='Learning rate value (default = 0.0001)')

args = parser.parse_args()

if not os.path.isfile(args.data):
	print("Error: File with the training set not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containing word embeddings not found.")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.output)) and not os.path.dirname(args.output) == "":
	print("Error: Output directory not found.")
	exit(code = 1)

print("Loading word embeddings...")
t_embeddings = text_embeddings.Embeddings()
t_embeddings.load_embeddings(args.embs, 100000, language = 'default', print_loading = True, skip_first_line = True)
vocabulary_size = len(t_embeddings.lang_vocabularies["default"])
embeddings = t_embeddings.lang_embeddings["default"].astype(np.float64)
embedding_size = t_embeddings.emb_sizes["default"]

print("Loading dataset...")
train_set = io_helper.load_csv_lines(args.data, delimiter = '\t')
train_wordpairs = [(x[0], x[1]) for x in train_set]
train_labels = [x[2] for x in train_set]
dist_labels = list(set(train_labels))
print("Preparing training examples...")
train_pairs, train_labels = data_shaper.prep_word_tuples(train_wordpairs, t_embeddings, "default", labels = train_labels)
train_labels, dl = data_shaper.prep_labels_one_hot_encoding(train_labels, dist_labels)
train_data = list(zip(train_pairs, train_labels))


# train params
num_maps = args.slice if args.slice else 5
lay_size = args.specsize if args.specsize else 100
lr = args.learningrate if args.learningrate else 0.0001
same_encoder = True
num_lays = 1
drp = 0.5
l2_reg_fac = 0.001
act = tf.nn.tanh
noise = 0
batch_size = 50

print("Defining the model...")
mapper_layers = [lay_size] * num_lays
model = wordpair_classifier.WordPairClassifier(embeddings, embedding_size, mapper_layers, same_mlp = same_encoder, bilinear_softmax = True, num_mappings = num_maps, activation = act, num_classes = len(dist_labels), noise_std = noise, dist_labels = dist_labels)
model.define_optimization(loss_functions.softmax_cross_entropy, l2_reg_fac, lr, loss_function_params = None)

print("Initializing a TensorFlow session...")
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

print("Training the model...")
coach = trainer.SimpleTrainer(model, session, build_feed_dict_func, None, None, None, None)
coach.train(train_data, batch_size, 10000, num_epochs_not_better_end = 20, epoch_diff_smaller_end = 0.001, print_batch_losses = False)

print("Serializing the model...")
ser_path = args.output
pickle.dump(model.get_model(session), open(args.output, "wb"))

print("My work here is done, ciao bella!")