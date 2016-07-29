from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import os
from random import shuffle
import time


class LabeledLineUtterance(object):
    def __init__(self, training_doc):
        self.training_doc = training_doc
        self.utterances = []

    def __iter__(self):
        with open(self.training_doc, 'r') as f:
            for utt_index, line in enumerate(f):
                yield TaggedDocument(words=line.split(), tags=['train' + '_%s' % utt_index])

    def to_array(self):
        with open(self.training_doc, 'r') as f:
            for index, line in enumerate(f):
                self.utterances.append(TaggedDocument(words=line.split(), tags=['train' + '_%s' % index]))
        return self.utterances

    def permute_utterances(self):
        shuffle(self.utterances)
        return self.utterances


def collate_files(dir):
    out = 'single_train.txt'
    with open(out, 'w') as out_file:
        for in_doc in [x for x in os.listdir(dir) if not x.startswith('.')]:
            with open(dir + in_doc, 'r') as in_file:
                for line in in_file:
                    out_file.write(line)


def how_many_utts(doc):
    return sum(1 for line in open(doc))


# training_dir = "../data/today/split/train/"
training_dir = "../data/today/split/train_small/"

test_dir = "../data/today/split/temp/"
train_source = 'single_train.txt'

# Make single training and test file
print "Collating training data to a single file..."
collate_files(training_dir)
print "Done.\n"
# Need to preprocess the text

print "Intiating and training doc2vec model..."
# Initiate model
train_sentences = LabeledLineUtterance(train_source)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
# Build the vocab table
model.build_vocab(train_sentences.to_array())
# train, permute training utterances in each epoch
epochs = range(5)
total_epochs = len(epochs)
for epoch in epochs:
    print "Epoch %i of %i..." % (epoch + 1, total_epochs)
    model.train(train_sentences.permute_utterances())
# Save model
print "Saving trained model to disk..."
model.save("doc2vec" + time.strftime("%d_%m_%Y_%H_%M_%S"))
print "Done.\n"

print "Inferring vectors for test data utterances..."
doc2vec_test_vectors = {}
test_index = 1
test_files = [x for x in os.listdir(test_dir) if not x.startswith(".")]
test_total = len(test_files)
for test_doc in test_files:
    print "Inferring vectors for test document %i of %i \t %s" % (test_index, test_total, test_doc)
    utt_vectors = []
    n_utts = how_many_utts(test_dir + test_doc)
    with open(test_dir + test_doc, 'r') as test:
        for utt in test:
            tokens = utt.split()
            utt_vectors.append(model.infer_vector(tokens)) # Consider adjusting params
    doc2vec_test_vectors[test_doc] = utt_vectors
    test_index += 1
print "Done.\n"