import tensorflow as tf

def softmax_cross_entropy(predictions, golds):
		losses = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=golds)
		loss = tf.reduce_mean(losses)
		return loss

def softmax_cross_entropy_micro_batches(predictions, golds, params):
	print("Defining micro-batched cross-entropy loss...")
	micro_batch_size, batch_size = params
	print("Micro-batch size: " + str(micro_batch_size))

	preds_unstacked = tf.unstack(predictions, num = batch_size)
	golds_unstacked = tf.unstack(golds, num = batch_size)

	if (len(preds_unstacked) % micro_batch_size != 0 or len(preds_unstacked) != len(golds_unstacked)):
		raise ValueError("Unexpected batch size, must be a multiplier of number of contrastive examples or num golds and predictions doesn't match!")
	
	loss = 0
	k = 0
	while k*micro_batch_size < len(preds_unstacked):
		print("Micro-batch iteration: " + str(k+1))
		preds_micro_batch = tf.nn.softmax(tf.stack(preds_unstacked[k*micro_batch_size : (k+1)*micro_batch_size]))
		golds_micro_batch = tf.nn.softmax(tf.stack(golds_unstacked[k*micro_batch_size : (k+1)*micro_batch_size]))
		loss += softmax_cross_entropy(preds_micro_batch, golds_micro_batch)
		k += 1
	return loss

def margin_based_loss(predictions, golds):
	return tf.reduce_sum(tf.maximum(tf.subtract(tf.constant(1.0, dtype = tf.float64), tf.multiply(predictions, golds)), 0.0))

def mse_loss(predictions, golds):
	return tf.reduce_sum(tf.square(tf.subtract(predictions, golds)))

def contrastive_loss(predictions, golds, params):
	print("Defining contrastive loss...")
	num_pos_pairs, num_neg_pairs, gamma = params
	preds_unstacked = tf.unstack(predictions)
	size = num_pos_pairs + num_neg_pairs
	if (len(preds_unstacked) % size != 0):
		raise ValueError("Unexpected batch size, must be a multiplier of number of contrastive examples!")
	
	loss = 0
	k = 0
	while k*size < len(preds_unstacked):
		pos_pairs = preds_unstacked[k*size : k*size + num_pos_pairs]
		print("Len of pos pair preds: " + str(len(pos_pairs)))
		neg_pairs = preds_unstacked[k*size + num_pos_pairs : (k+1) * size]
		print("Len of neg pair preds: " + str(len(neg_pairs)))
		for p in pos_pairs:
			for n in neg_pairs:
				loss += tf.maximum(tf.constant(0.0, dtype = tf.float64), gamma - (p - n))
		k += 1
	return loss

def contrastive_loss_nonbinary(predictions, golds, params):
	print("Defining contrastive loss...")
	num_pos_pairs, num_neg_pairs, mean_square_error, batch_size = params
	preds_unstacked = tf.unstack(predictions, num = batch_size)
	golds_unstacked = tf.unstack(golds, num = batch_size)

	size = num_pos_pairs + num_neg_pairs
	if (len(preds_unstacked) % size != 0 or len(preds_unstacked) != len(golds_unstacked)):
		raise ValueError("Unexpected batch size, must be a multiplier of number of contrastive examples or num golds and predictions doesn't match!")
	
	loss = 0
	k = 0
	while k*size < len(preds_unstacked):
		print("Micro-batch iteration: " + str(k+1))
		pos_pairs = preds_unstacked[k*size : k*size + num_pos_pairs]
		pos_golds = golds_unstacked[k*size : k*size + num_pos_pairs]
		print("Len of pos pair preds: " + str(len(pos_pairs)))

		neg_pairs = preds_unstacked[k*size + num_pos_pairs : (k+1) * size]
		neg_golds = golds_unstacked[k*size + num_pos_pairs : (k+1) * size]
		print("Len of neg pair preds: " + str(len(neg_pairs)))

		for i in range(len(pos_pairs)):
			for j in range(len(neg_pairs)):
				if mean_square_error:
					if k == 0 and i == 0 and j == 0: 
						print("MSE NCE loss for pair...")
					loss += tf.square((pos_golds[i] - neg_golds[j]) - (pos_pairs[i] - neg_pairs[j]))
				else:
					if k == 0 and i == 0 and j == 0: 
						print("Hinge, margin loss for pair...")
					loss += tf.maximum(tf.constant(0.0, dtype = tf.float64), (pos_golds[i] - neg_golds[j]) - (pos_pairs[i] - neg_pairs[j]))
		k += 1
	return loss
		
	

###### Not differentiable
#def misclassification_count(predictions_scores, gold_label_scores, l2_loss, l2_reg_factor):
#		preds = tf.argmax(predictions_scores, 1, name="predictions")
#		gs = tf.argmax(gold_label_scores, 1)
#		loss = tf.reduce_sum(tf.cast(tf.not_equal(preds, gs), "float")) #+ l2_reg_factor * l2_loss
#		return loss








################## OLD #####################
#def multi_class_hinge_loss(prediction_scores, gold_scores, l2_loss, l2_reg_factor):
#	loss = tf.contrib.losses.hinge_loss(prediction_scores, gold_scores) + l2_reg_factor * l2_loss
#	return loss

#def kb_embed_simple_loss(prediction_scores, gold_scores, l2_loss, l2_reg_factor):
#	scores = prediction_scores[0]
#	loss = -1 * tf.reduce_sum(tf.multiply(scores, gold_scores)) + l2_reg_factor * l2_loss
#	return loss

## this assumes that batches contain full positive-fake pairs, i.e., that each true triple is immediately followed by all its corresponding corrupted triples
#def kb_embed_margin_ranking_loss(prediction_scores, gold_scores, l2_loss, l2_reg_factor):
#	triple_scores = prediction_scores[0]
#	num_corrupted = prediction_scores[1]
	
#	triple_scores = tf.reshape(triple_scores, [-1, num_corrupted + 1])
#	slices = tf_extensions.slice_matrix(triple_scores, [1], col_or_row = "col")
#	positives = tf.tile(slices[0], [1, num_corrupted])
#	diff = tf.add(tf.subtract(slices[1], positives), 1)
#	pure_loss = tf.reduce_sum(tf.maximum(diff, 0))
	
#	loss = pure_loss + l2_reg_factor * l2_loss
#	return loss
	
		
	
	
	
