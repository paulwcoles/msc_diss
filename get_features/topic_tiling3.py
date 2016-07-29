import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import csr_matrix
from nltk.metrics.segmentation import windowdiff
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle
import time
from sys import argv, exit
import pickle

# PARAMETERS
# CountVectorizer
max_df = 0.95
min_df = 2
# TfidfTransformer
norm = 'l2'
sublinear_tf = False
# LDA
n_topics = 100
n_iterations = 50
doc_topic_prior = 50/n_topics # alpha
topic_word_prior = 0.01 # eta
max_iter = 10
n_features = None
n_tiling_iterations = 2
# Doc2Vec
train_iter = 10
min_count = 1
doc2vec_window = 8
doc2vec_size = 100
sample = 1e-4
negative = 5
# TopicTiling
window_size = 2 # w in literature
# seg_method = 'threshold'
seg_method = 91
n_tiling_iterations = 5 # Set to 1 for d=false
# Quick parameters
# n_features = 100
# n_topics = 30
# n_top_words = 20
# n_iterations = 5
# train_iter = 2

# OPERATION MODES
switch = argv[1]
if switch not in ['lda', 'doc2vec', 'tfidf', 'uniform']:
    print "Invalid mode."
    exit(1)
if switch == 'uniform':
    uniform = True

mode = 'evaluate'
# mode = 'resolve'
play = False


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


def load_corpus(docs_dir):
    corpus = {}
    for subdir, dirs, files in os.walk(docs_dir):
        for f in files:
            if not f.startswith("."):
                with open(docs_dir + f, 'r') as doc:
                    corpus[f] = doc.read().decode("utf-8", "replace").lower()
    return corpus


def convert_to_windowed(in_matrix, window_size):
    out_matrix = csr_matrix(in_matrix.get_shape(), dtype=int)
    out_matrix = out_matrix.tolil()
    n_vectors = out_matrix.get_shape()[0]
    # Relative to current utt vector, window starts one position forward and back
    window_range = range(1, window_size + 1)
    for i in range(n_vectors):
        out_matrix[i] += in_matrix[i]
        for j in window_range:
            # Window left if possible
            if i - j >= 0:
                out_matrix[i] += (in_matrix[i - j])
            # Window right if possible
            if i + j < n_vectors:
                out_matrix[i] += in_matrix[i + j]
    return out_matrix


def make_cosine_list(matrix):
    # Returns list: nth index in list is cosine sim of nth and nth + 1 utt in matrix (final utt cannot have cosine score)
    cosines = []
    if switch == 'tfidf': # matrix type: scipy.sparse.lil.lil_matrix
        n_vectors = matrix.shape[0]
        for i in xrange(n_vectors - 1):
            cosines.append(cosine_similarity(matrix[i:i+1], matrix[i+1:i+2])[0][0])
    elif switch == 'lda' or switch == 'doc2vec': # matrix type: list
        n_vectors = len(matrix)
        for i in xrange(n_vectors - 1):
            cosines.append(cosine_similarity(matrix[i].reshape(1, -1), matrix[i+1].reshape(1, -1))[0][0])
    # elif switch == 'doc2vec':
    #     print type(matrix)
    return cosines


def score_depth(cosines, verbose):
    # Find the minima as indicies to original cosines
    minima = [None]
    # Only 2nd to n-1th element could be minimum
    for index, score in enumerate(cosines[1:-1]):
        current_score = cosines[index]
        if cosines[index - 1] > current_score and cosines[index + 1] > current_score:
            minima.append(index + 1)
        else:
            minima.append(None)
    minima.append(None)
    # For each minimum, compute depth score
    depth_scores = []
    score_count = 0
    zero_count = 0
    for minimum in minima:
        if minimum is not None:
            # Search backwards from each minimum: take left slice (until, and not inc., minimum) in reverse order
            back_portion = (cosines[:minimum])[::-1]
            b_max, distance = find_max(back_portion)
            # Search forward from each minimum: take right slice (from, and not inc., minimum)
            forward_portion = cosines[minimum + 1:]
            f_max, distance = find_max(forward_portion)
            depth_score = 0.5 * ((b_max - cosines[minimum]) + (f_max - cosines[minimum]))
            depth_scores.append(depth_score)
            score_count += 1
            if verbose:
                print "minimum\t" + str(minimum)
                print "Backward portion:\t" + str(back_portion)
                print "b_max\t" + str(b_max) + "\tdistance\t" + str(distance)
                print "Forward portion:\t" + str(forward_portion)
                print "f_max\t" + str(f_max) + "\tdistance\t" + str(distance)
                print "Depth score:\t" + str(depth_score) + '\n'
        else:
            depth_scores.append(0.0)
            zero_count += 1
    return np.array(depth_scores), score_count, zero_count


def find_max(portion):
    distance = 1
    current_max = portion[0]
    for possible_max_index in range(len(portion) - 1):
        current_max = portion[possible_max_index]
        next_candidate = portion[possible_max_index + 1]
        # Keep searching while next possible max is increasing
        if next_candidate > current_max:
            current_max = next_candidate
            distance += 1
        # Else stop
        else:
            break
    return current_max, distance


def parse_annotation(annotation_dir):
    gold_boundaries_set = {}
    for f in os.listdir(annotation_dir):
        gold_boundaries = []
        gold_count = 0
        line_count = 0
        non_blank = 0
        with open(annotation_dir + f, 'r') as raw_annotation:
            for line in raw_annotation:
                line_count += 1
                if line != "":
                    non_blank += 1
                    if line.startswith('# '):
                        gold_boundaries.append("BREAK")
                        gold_count += 1
                    else:
                        gold_boundaries.append(None)
        gold_boundaries_set[f] = gold_boundaries
    return gold_boundaries_set, gold_count


def reverse_parse(doc_name, predicted_boundaries):
    out_dir = "../data/today/split/predicted/" + timestamp + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(test_dir + doc_name, 'r') as in_file:
        in_file_list = list(in_file)
        with open(out_dir + doc_name, 'w') as out_file:
            for index, entry in enumerate(predicted_boundaries):
                current_line = in_file_list[index]
                if entry is None or current_line.startswith("# "):
                    out_file.write(current_line)
                else:
                    out_file.write("# " + current_line)


def get_utt_timing(doc_name):
    doc_name = doc_name.replace("_parsed","")
    doc_name = doc_name.replace(".txt",".ctm")
    start_times = []
    with open(timings_dir + doc_name, 'r') as raw:
        for line in raw:
            split_line = line.split()
            times = (split_line[0].split('-'))[4]
            start_time = int(times.split(':')[0])
            if start_time not in start_times:
                start_times.append(start_time)
    return start_times


def plot_doc(cosines):
    import matplotlib.pyplot as plt
    plt.plot(cosines)
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Utterance Pair')
    plt.show()


def how_many_utts(doc):
    return sum(1 for line in open(doc))


def how_many_words(doc):
    return len(set(w.lower() for w in open(doc).read().split()))


def collate_files(dir, window):
    out = "../data/today/split/doc2vec_train/" + "single_train_" + str(window) + ".txt"
    if not os.path.isfile(out):
        window_range = range(1, window + 1)
        window_range_reverse = window_range[::-1]
        with open(out, 'w') as out_file:
            for in_doc in [x for x in os.listdir(dir) if not x.startswith('.')]:
                with open(dir + in_doc, 'r') as in_file:
                    in_file_list = list(in_file)
                    n_in_file_utts = len(in_file_list)
                    for i, utt in enumerate(in_file_list):
                        windowed = []
                        # Window left
                        for j in window_range_reverse:
                            if i - j >= 0:
                                windowed.append(in_file_list[i - j])
                        windowed.append(utt)
                        # Window right
                        for j in window_range:
                            if i + j < n_in_file_utts:
                                windowed.append(in_file_list[i + j])
                        # Write out
                        out_line_temp = " ".join([x.strip("\n") for x in windowed[:-1]])
                        out_line = out_line_temp + " " + windowed[-1]
                        out_file.write(out_line)


def get_depth_scores(doc_name, cosines):
     # Compute cosines, depth score
    depth_scores, score_count, zero_count = score_depth(cosines, verbose=False)
    non_zero_depth_scores = [x for x in depth_scores if x != 0.0]
    predicted_boundaries, boundary_count = [None] * len(depth_scores), 0
    if seg_method == 'threshold':
        threshold = (np.mean(non_zero_depth_scores) - np.std(non_zero_depth_scores)) / 2
        for index, depth_score in enumerate(depth_scores):
            if depth_score > threshold and depth_score != 0.0:
                predicted_boundaries.insert(index, "BREAK")
                boundary_count += 1
    elif type(seg_method) is int:
        # Sort depth scores descending, take slice (up to desired number of boundaries)
        winning_indices = (sorted(range(len(depth_scores)), key=lambda k: depth_scores[k])[::-1])[:seg_method]
        for index in winning_indices: # Insert boundary at top n indices
            predicted_boundaries[index] = "BREAK"
            boundary_count += 1
    predicted_boundaries.append(None) # There is one more boundary than there are depth scores
    reverse_parse(doc_name, predicted_boundaries) # Store predictions
    if mode == 'evaluate': # Perform evaluation
        evaluate(predicted_boundaries, doc_name)


def evaluate(predicted_boundaries, doc_name):
    evaluation = windowdiff(predicted_boundaries, gold_boundaries_set[doc_name], 3, boundary="BREAK",
                        weighted=False)
    print "Window Diff Score:\t %f\n" % evaluation
    evaluations[doc_name] = evaluation
    with open(log_dir + timestamp, 'a') as log:
        log.write("\n" + doc_name + "\t"*3 + str(evaluation))

def train_doc2vec():
    collate_files(training_dir, window_size)
    train_utts = LabeledLineUtterance("single_train_" + str(window_size) + ".txt")
    doc2vec_model = Doc2Vec(iter=train_iter, min_count=min_count, window=doc2vec_window, size=doc2vec_size,
                            sample=sample, negative=negative, workers=8) # Initiate model
    doc2vec_model.build_vocab(train_utts.to_array()) # Build the vocab table
    epochs = range(train_iter) # train, permute training utterances in each epoch
    total_epochs = len(epochs)
    for epoch in epochs:
        print "Epoch %i of %i..." % (epoch + 1, total_epochs)
        doc2vec_model.train(train_utts.permute_utterances())
    doc2vec_model.save("../data/doc2vec_models/" + timestamp) # Save model
    return doc2vec_model

def test_doc2vec():
    test_index = 1
    test_files = [x for x in test_dataset.keys() if not x.startswith(".")]
    test_total = len(test_files)
    for test_doc in test_files:
        print "Inferring vectors for test document %i of %i \t %s" % (test_index, test_total, test_doc)
        utt_vectors = []
        n_utts = how_many_utts(test_dir + test_doc)
        with open(test_dir + test_doc, 'r') as test:
            for utt in test:
                tokens = utt.split()
                utt_vectors.append(doc2vec_model.infer_vector(tokens))
        test_doc_cosines = make_cosine_list(utt_vectors)
        get_depth_scores(test_doc, test_doc_cosines)
        test_index += 1


def make_word_matrices():
    try:
        word_wise_test_dataset = pickle.load(open("./word_wise_test_data/word_wise_test_dataset", 'r'))
        test_corpus_topic_counts = pickle.load(open("./word_wise_test_data/test_corpus_topic_counts", 'r'))
        print "Opened existing utterance-wise windowed count matrices."
        virgin = False
    except:
        print "No utterance-wise windowed count matrices existing. Creating new..."
        word_wise_test_dataset = {}
        test_corpus_topic_counts = {}
        n_test_docs = len(test_dataset)
        test_doc_index = 1
        for doc_name, content in test_dataset.iteritems(): # Make word-wise count matrices
            utts = [x for x in content.split('\n') if x != ""]
            utt_matrices = []
            test_corpus_topic_counts[doc_name] = [] # Each doc is list...
            n_utts = len(utts)
            for utt_index, utt in enumerate(utts):
                print "Making utterance-wise windowed count matrices: \t test doc %i of %i \t utterance %i of %i" % \
                      (test_doc_index, n_test_docs, utt_index + 1, n_utts)
                split_utt = utt.split()
                tf_test = count_vectorizer.transform(split_utt) # Matrix over words in utt
                windowed_count_matrix = convert_to_windowed(tf_test, window_size)
                utt_matrices.append(windowed_count_matrix)
                test_corpus_topic_counts[doc_name].append(None) #... each utt of doc as list of n lists, n = words in utt
                test_corpus_topic_counts[doc_name][utt_index] = [[] for i in xrange(windowed_count_matrix.get_shape()[0])]
            word_wise_test_dataset[doc_name] = utt_matrices
            test_doc_index += 1
        pickle.dump(word_wise_test_dataset, open("./word_wise_test_data/word_wise_test_dataset", 'w'))
        pickle.dump(test_corpus_topic_counts, open("./word_wise_test_data/test_corpus_topic_counts", 'w'))
        virgin = True
    return word_wise_test_dataset, test_corpus_topic_counts, virgin

def blank_word_lists(test_corpus_topic_counts):
    new_dict = {}
    for doc_name, doc_vector in test_corpus_topic_counts.iteritems():
        new_doc_vector = []
        for utt_index, utt_vector in enumerate(doc_vector):
            new_doc_vector.append([])
            for word_index, word_list in enumerate(utt_vector):
                new_doc_vector[utt_index].append([])
        new_dict[doc_name] = new_doc_vector
    return new_dict


def lda_train_test(n_tiling_iterations, tf_matrix):
    word_wise_test_dataset, test_corpus_topic_counts, virgin = make_word_matrices()
    if not virgin:
        test_corpus_topic_counts = blank_word_lists(test_corpus_topic_counts)
    tiling_range = range(n_tiling_iterations)
    for i in tiling_range:     # Store topic assignment per word, then retrain
        print "Fitting LDA model..."
        lda.fit_transform(tf_matrix)
        print "Done."
        doc_count = 1
        n_docs = len(test_dataset)
        for doc_name, utt_matrices in word_wise_test_dataset.iteritems():
            for utt_index, utt_matrix in enumerate(utt_matrices):
                print "Topic assignment iteration: %i of %i\t\tTest doc %i of %i: %s\t\tUtterance %i of %i..." % \
                      (i + 1, n_tiling_iterations, doc_count, n_docs, doc_name, utt_index + 1, len(utt_matrices))
                utt_lda = lda.transform(utt_matrix)
                for word_index, word_vector in enumerate(utt_lda):
                    test_corpus_topic_counts[doc_name][utt_index][word_index].append(np.argmax(word_vector))
    # Make topic vectors: count mode topic for each word.
    print "Making topic vectors from mode topic assignments, computing cosine similarity..."
    doc_count = 1
    for doc_name, doc_topic_counts in test_corpus_topic_counts.iteritems():
        print "Test doc %d of %d..." % (doc_count, n_docs)
        doc_topic_vector_list = []
        for utt_topic_counts in doc_topic_counts:
            utt_topic_vector = np.zeros(n_topics, dtype="int")
            for word_topic_assignments in utt_topic_counts:
                mode_topic = max(set(word_topic_assignments), key=word_topic_assignments.count)
                utt_topic_vector[mode_topic] += 1
            doc_topic_vector_list.append(utt_topic_vector)
        cosines = make_cosine_list(doc_topic_vector_list) # Compute cosine for adjacent utts.
        get_depth_scores(doc_name, cosines) # Score depth, predict boundaries.
        doc_count += 1


if __name__ == "__main__":
    timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
    training_dir = "../data/today/split/train/"
    # training_dir = "../data/today/split/train_small/"
    test_dir = "../data/today/split/test/"
    # test_dir = "../data/today/split/temp/"
    timings_dir = "../data/today/split/timings/"
    log_dir = "../logs/"
    with open(log_dir + timestamp, 'wb') as log:
        log.write(timestamp + "\n" + "-" * 20 + " PARAMETERS " + "-" * 20 + "Train:" + "\t"*4 + training_dir + "\nTest:" +
                  "\t"*4 + test_dir + "\nMode:" + "\t"*4 + switch + "\nTiling Window:" + "\t"*4 + str(window_size))
        if switch == "tfidf" or "lda":
            log.write("\n\nmax_df:" + "\t"*4 + str(max_df) + "\nmin_df:" + "\t"*4 + str(min_df) + "\nnorm:" + "\t"*4
                      + str(norm) + "\nsublinear_tf:" + "\t"*4 + str(sublinear_tf))
        if switch == "lda":
            pass
        log.write("\n\n" + "-" * 20 + "  RESULTS  " + "-" * 20 + "\n")

    print "-" * 28 + "  TopicTiling  " + "-" * 28
    print "Start:\t %s" % timestamp
    print "Mode: \t %s\n" % (switch)
    print "-" * 30 + "  Train  " + "-" * 30
    print "Loading from training and test datasets..."
    training_dataset = load_corpus(training_dir)
    gold_boundaries_set, gold_count = parse_annotation(annotation_dir=test_dir)
    test_dataset = load_corpus(test_dir)
    count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', analyzer='word')
    tf_matrix = count_vectorizer.fit_transform(training_dataset.values())
    tfidf_transformer = TfidfTransformer(norm=norm, use_idf=True, smooth_idf=True, sublinear_tf=sublinear_tf)
    tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
    evaluations = {}
    topic_timings = {}

    if uniform:
        # Baseline performance: segment test docs into n uniform segments
        if type(seg_method) is int:
            for doc_name, content in test_dataset.iteritems():
                utts = [x for x in content.split('\n') if x != ""]
                n_utts = len(utts)
                uniform_boundary_gap = n_utts/seg_method
                woodblock_predictions = []
                for i in xrange(n_utts):
                    if i % uniform_boundary_gap == 0:
                        woodblock_predictions.append("BREAK")
                    else:
                        woodblock_predictions.append(None)
                print doc_name
                evaluate(woodblock_predictions, doc_name)
        else:
            print "Uniform segmentation requires n segments method (not threshold)."
            exit(1)


    if switch == 'doc2vec':
        print "Initiating and training doc2vec model..."
        doc2vec_model = train_doc2vec()
        print "-" * 30 + "  Test  " + "-" * 30
        test_doc2vec()

    elif switch == 'tfidf':
        print "-" * 30 + "  Test  " + "-" * 30
        test_doc_i = 1
        n_test_docs = len(test_dataset)
        for doc_name, content in test_dataset.iteritems():
            print "Test doc %i of %i..." % (test_doc_i, n_test_docs)
            utts = [x for x in content.split('\n') if x != ""]
            utt_vectors = []
            n_utts = len(utts)
            tf_test = count_vectorizer.transform(utts)
            windowed_count_matrix = convert_to_windowed(tf_test, window_size)
            cosines = make_cosine_list(windowed_count_matrix)
            get_depth_scores(doc_name, cosines)
            test_doc_i += 1

    elif switch == 'lda':
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=n_iterations,
                                           learning_method='online', learning_offset=50.,
                                           random_state=0, doc_topic_prior=doc_topic_prior,
                                           topic_word_prior=topic_word_prior)
        lda_train_test(n_tiling_iterations, tf_matrix)
    print "-" * 30 + "  End  " + "-" * 30


    #     if mode == 'resolve':
    #         print "Resolving topic timings..."
    #         utt_start_times = get_utt_timing(doc_name)
    #         topic_start_times = [utt_start_times[0]]  # Initialise with start time of first utterance
    #         for index in xrange(len(predicted_boundaries)):
    #             if predicted_boundaries[index] is not None:
    #                 topic_start_times.append(utt_start_times[index])
    #         prg_name = doc_name.replace("_parsed","")
    #         prg_name = prg_name.replace(".txt",".wav")
    #         topic_timings[prg_name] = topic_start_times
    #     elif mode == 'resolve' and play:
    #         import pickle
    #         pickle.dump(topic_timings, open("topic_timings", "wb"))