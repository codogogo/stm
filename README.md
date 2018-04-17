# Specialization Tensor Model  
Specialization Tensor Model (STM) is a simple and effective feed-forward neural model for classifying (lexico-semantic) relations for pairs of words, given their word embedding as input

## Running the code

There are to Python scripts that you can run: *stm-train.py* and *stm-predict.py*. The former trains the model using the provided training data, whereas the latter one makes predictions for arbitrary word pairs, using a model pre-trained with the first script. 

By running each of the scripts with the option *-h*, you can see the complete lists of their mandatory and optional arguments.  

### Training the relation classifier 

The training script *stm-train.py* traines the relation classification model, requiring the following arguments: 

1. "data" is the path to the file containing the list of word pairs (with relation labels), to be used to train the classifier;  
2. "embs" is the path to the file containing the pre-trained word embeddings;
3. "output" is the path to which the serialize the classifier (i.e., the model) once it's trained. 

Additionally, you can configure one of the following optional arguments: 

- *\-\-slice* defines the number of slices of the specialization tensor (the value must be an integer, default value is K = 5);
- *\-s* (or *\-\-specsize*) is the dimension of the specialized vectors (the value must be an integer, default value is n = 100); 
- *\-l* (or *\-\-learningrate*) is the initial value of the learning rate for the Adam optimization algorithm (default values is 0.0001).

### Predicting relations with a trained classifier 

Once you have trained the classifier using the script *stm-train.py*, you can use the trained model to prediction relations for arbitrary word pairs using the script *stm-predict.py*. The prediction script requires the following arguments: 

1. "data" is the path to the file containing the list of word pairs for which you want to predict the (lexico-semantic) relation;
2. "model" is the path to the serialized (pickled) model file, previously trained and stored using *stm-train.py*;
3. "embs" is the path to the file containing the pre-trained word embeddings;
4. "preds" is the path to which you want to store the predictions for the word pairs from the "data" file. 

### Prerequisites

- *numpy* (tested with version 1.12.1)
- *scipy* (tested with version 0.19.0)
- *tensorflow* (tested with version 1.3.0). 

## Data 

The input data file for *stm-train.py* needs to contain one word pair with a relation label per line. The words and the label need to be separated with a tabular: *word1\tword2\tlabel*. You can see the examples of the training files in the folder *data*. The data file used as input for the prediction script *stm-predict.py* needs not necessarily contain the label (but it may), i.e., each line can be in the format: *word1\tword2*. 

The datasets used for training and evaluation in the paper (WN-LS and CogaLex-V dataset; see reference below) can be found in the *data* folder. 

## Referencing

If you're using STM in your work, please cite the following paper: 

```
@InProceedings{glavavs-vulic:2018:NAACL-HLT,
  author    = {Glava\v{s}, Goran  and  Vulic\'{c}, Ivan},
  title     = {Discriminating Between Lexico-Semantic Relations with the Specialization Tensor Model},
  booktitle = {Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {in print}
}

```

 
