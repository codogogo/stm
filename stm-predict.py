import argparse
import os
import tensorflow as tf
import numpy as np
import pickle
from helpers import io_helper
from helpers import data_shaper
from embeddings import text_embeddings
from models import wordpair_classifier
from ml import loss_functions

def build_feed_dict_func(model, data, config = None, predict = False):
	x_pairs, y = zip(*data)
	x1s = [x[0] for x in x_pairs]
	x2s = [x[1] for x in x_pairs]
	drp = 0.5

	fd = model.get_feed_dict(x1s, x2s, None if predict else y, 1.0 if predict else drp, not predict)
	return fd, y

parser = argparse.ArgumentParser(description='Predicts a lexico-semantic relation between pairs of words, using a previously trained classification model.')
parser.add_argument('data', help='A path to the file containing prediction examples (test set). Each test example should have the format: "first_word TAB second_word". Pairs containing words that are not in the vocabulary of the input distributional embeddings will be skipped.')
parser.add_argument('model', help='A path to the file containing the serialized model.')
parser.add_argument('embs', help='A path to the file containing pre-trained word embeddings')
parser.add_argument('preds', help='A file path to which to store predictions of the model for the test set pairs.')

args = parser.parse_args()

if not os.path.isfile(args.data):
	print("Error: File with the test set not found.")
	exit(code = 1)

if not os.path.isfile(args.model):
	print("Error: File containig a serialized model not found.")
	exit(code = 1)

if not os.path.isfile(args.embs):
	print("Error: File containig pretrained word embeddings not found.")
	exit(code = 1)

if args.preds is not None and not os.path.isdir(os.path.dirname(args.preds)) and not os.path.dirname(args.preds) == "":
	print("Error: Predictions output directory not found.")
	exit(code = 1)


print("Loading word embeddings...")
t_embeddings = text_embeddings.Embeddings()
t_embeddings.load_embeddings(args.embs, 100000, language = 'default', print_loading = True, skip_first_line = True)
vocabulary_size = len(t_embeddings.lang_vocabularies["default"])
embeddings = t_embeddings.lang_embeddings["default"].astype(np.float64)
embedding_size = t_embeddings.emb_sizes["default"]
t_embeddings.inverse_vocabularies()

print("Loading model...")
hyperparams, vars = pickle.load(open(args.model, "rb" ))

same_mlp = hyperparams[0]
bilinear_softmax = hyperparams[1]
mlp_hidden_layer_sizes = hyperparams[2]
num_mlps = hyperparams[3]
embedding_size = hyperparams[5]
dist_labels = hyperparams[6]
act = tf.nn.tanh
noise = 0

model = wordpair_classifier.WordPairClassifier(embeddings, embedding_size, mlp_hidden_layer_sizes, same_mlp = same_mlp, bilinear_softmax = bilinear_softmax, num_mappings = num_mlps, activation = act, num_classes = len(dist_labels), noise_std = noise)
#model.define_optimization(loss_functions.softmax_cross_entropy, l2_reg_fac, lr, loss_function_params = None)

print("Initializing tensorflow session...")
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

model.set_variable_values(session, vars)


###### preparing test set for predictions #######
print("Loading dataset...")
predict_set = io_helper.load_csv_lines(args.data, delimiter = '\t')
predict_wordpairs = [(x[0], x[1]) for x in predict_set]
#predict_labels = [x[2] for x in predict_set]
print("Preparing prediction examples...")
predict_pairs = data_shaper.prep_word_tuples(predict_wordpairs, t_embeddings, "default", labels = None)
#predict_labels, dl = data_shaper.prep_labels_one_hot_encoding(predict_labels, dist_labels)
predict_data = list(zip(predict_pairs, [None]*len(predict_pairs)))
	
###### predicting and evaluating ################
print("Computing predictions...")
preds = model.preds_raw.eval(session = session, feed_dict = build_feed_dict_func(model, predict_data, predict = True)[0])
pred_labels = [dist_labels[np.argmax(p)] for p in preds]

if args.preds is not None:
	print("Writing predictions to file...")
	to_write = list(zip([t_embeddings.get_word_from_index(x[0], lang = "default") for x in predict_pairs], [t_embeddings.get_word_from_index(x[1], lang = "default") for x in predict_pairs], pred_labels))
	io_helper.write_list_tuples_separated(args.preds, to_write)

print("My work here is done, ciao bella!")