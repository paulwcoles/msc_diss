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


# PARAMETERS
# CountVectorizer
max_df = 1.0
min_df = 1
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
# Doc2Vec
train_iter = 10
min_count = 1
doc2vec_window = 8
doc2vec_size = 100
sample = 1e-4
negative = 5
# TopicTiling
window_size = 3
# seg_method = 'threshold'
seg_method = 20
# Quick parameters
n_features = 100
n_topics = 30
n_top_words = 20
n_iterations = 5
train_iter = 2

# OPERATION MODES
# switch = 'tfidf'
switch = 'lda'
# switch = 'doc2vec'
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
    # Returns list: nth index in list is cosine sim of nth and nth + 1 utt in matrix
    cosines = []
    if switch == 'doc2vec' or switch == 'lda':
        # Matrix is numpy array
        n_vectors = len(matrix)
        for i in xrange(n_vectors - 1):
            cosines.append(cosine_similarity(matrix[i].reshape(1, -1), matrix[i+1].reshape(1, -1)))
    elif switch == 'tfidf':
        n_vectors = matrix.shape[0]
        # Iterate over each utt in doc_matrix, score with subsequent utt (final utt cannot have cosine score)
        for i in xrange(n_vectors - 1):
        # Index the Sci-Py matrix by slice to extract vectors
            cosines.append(cosine_similarity(matrix[i:i+1], matrix[i+1:i+2])[0][0])
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


def make_topic_vectors(utts):
    topic_vectors = np.empty((len(utts), n_topics), dtype="int")
    n_tiling_iterations = 2
    tiling_range = range(n_tiling_iterations)
    utt_count = len(utts)
    for index, utt in enumerate(utts):
        print "Test doc %i of %i:\t%s\t\tUtterance %i of %i..." % (doc_count, n_docs, doc_name, index + 1,
                                                                 utt_count)
        topic_vector = np.empty(n_topics, dtype="int")
        # Matrix over words in utt
        split_utt = utt.split()
        tf_test = count_vectorizer.transform(split_utt)
        windowed_count_matrix = convert_to_windowed(tf_test, window_size)
        # Empty list for each word in utt
        word_topic_assignments = [[]] * (windowed_count_matrix.get_shape()[0])
        # At each inference iteration, make fresh LDA matrix of words in utt...
        utt_lda = lda.transform(windowed_count_matrix)
        for i in tiling_range:
            # utt_lda = lda.transform(windowed_count_matrix)
            # ... and for each word, store the index of its most probable topic
            for j, word in enumerate(utt_lda):
                word_topic_assignments[j].append(np.argmax(word))
        # After completing inference iterations, for each word choose the mode topic assignment...
        for assignment_set in word_topic_assignments:
            topic_of_word = max(set(assignment_set), key=assignment_set.count)
            # ... and increment the count for that topic in the utterance
            topic_vector[topic_of_word] += 1
        topic_vectors[index] = topic_vector
    return topic_vectors


## HELPER FUNCTIONS
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

if __name__ == "__main__":
    timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
    training_dir = "../data/today/split/train/"
    training_dir = "../data/today/split/train_small/"
    # test_dir = "../data/today/split/test/"
    test_dir = "../data/today/split/temp/"
    timings_dir = "../data/today/split/timings/"
    log_dir = "../logs/"
    with open(log_dir + timestamp, 'wb') as log:
        log.write(timestamp + "\n" + "-" * 20 + " PARAMETERS " + "-" * 20 + "Train:" + "\t"*4 + training_dir + "\nTest:" +
                  "\t"*4 + test_dir + "\n\nMode:" + "\t"*4 + switch + "\nTiling Window:" + "\t"*4 + str(window_size))
        if switch == "tfidf" or "lda":
            log.write("\n\nmax_df:" + "\t"*4 + str(max_df) + "\nmin_df:" + "\t"*4 + str(min_df) + "\nnorm:" + "\t"*4
                      + str(norm) + "\nsublinear_tf:" + "\t"*4 + str(sublinear_tf))
        if switch == "lda":
            pass
        log.write("\n\n" + "-" * 20 + "  RESULTS  " + "-" * 20 + "\n")



    print "-" * 28 + "  TopicTiling  " + "-" * 28
    print "Start:\t %s" % timestamp
    print "Mode: \t %s\n" % (switch)

    # # # TRAIN # # #
    print "-" * 30 + "  Train  " + "-" * 30
    n_tiling_iterations = 2
    tiling_range = range(n_tiling_iterations)

    for i in t
    if switch != 'doc2vec':
        # Load training corpus.
        print "Loading training dataset..."
        training_dataset = load_corpus(training_dir)
        print "Done.\n"
        # Make tf matrix
        print "Extracting tf features (raw counts)..."
        count_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english', analyzer='word')
        tf_matrix = count_vectorizer.fit_transform(training_dataset.values())
        print "Done.\n"

        # Weight into tf-idf space.
        print "Weighting into tf-idf space, fitting model to training corpus..."
        tfidf_transformer = TfidfTransformer(norm=norm, use_idf=True, smooth_idf=True, sublinear_tf=sublinear_tf)
        tfidf_matrix = tfidf_transformer.fit_transform(tf_matrix)
        print "Done.\n"

        # Use tf-idf matrix to fit LDA model
        if switch == 'lda':
           print "Fitting LDA models with tf features:\t %d iterations..." % (n_iterations)
           lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=n_iterations,
                                           learning_method='online', learning_offset=50.,
                                           random_state=0, doc_topic_prior=doc_topic_prior,
                                           topic_word_prior=topic_word_prior)
           lda_matrix = lda.fit_transform(tf_matrix)
           print "Done.\n"

    else:
        # Make / open single training file
        print "Initiating and training doc2vec model..."
        collate_files(training_dir, window_size)
        train_utts = LabeledLineUtterance("single_train_" + str(window_size) + ".txt")
        # Initiate model
        doc2vec_model = Doc2Vec(iter=train_iter, min_count=min_count, window=doc2vec_window, size=doc2vec_size,
                                sample=sample, negative=negative, workers=8)
        # Build the vocab table
        doc2vec_model.build_vocab(train_utts.to_array())
        # train, permute training utterances in each epoch
        epochs = range(train_iter)
        total_epochs = len(epochs)
        for epoch in epochs:
            print "Epoch %i of %i..." % (epoch + 1, total_epochs)
            doc2vec_model.train(train_utts.permute_utterances())
        # Save model
        print "Saving trained model to disk..."
        doc2vec_model.save("../data/doc2vec_models/" + timestamp)
        print "Done.\n"


    # # # TEST # # #
    print "-" * 30 + "  Test  " + "-" * 30
    print "Parsing human annotations..."
    gold_boundaries_set, gold_count = parse_annotation(annotation_dir=test_dir)
    test_dataset = load_corpus(test_dir)
    evaluations = {}
    topic_timings = {}
    print "Done.\n"

    print "Making test document utterance vectors..."
    test_vectors = {}
    if switch != 'doc2vec':
        doc_count = 1
        n_docs = len(test_dataset)
        print "Done.\n"
        for doc_name, content in test_dataset.iteritems():
            # Split doc string by utterance
            utts = [x for x in content.split('\n') if x != ""]
            if switch == 'lda':
                print "Computing cosine similarity for adjacent utterances in LDA matrices..."
                topic_vectors = make_topic_vectors(utts)
                test_doc_cosines = make_cosine_list(topic_vectors)
            else: # switch == tfidf
                print "Computing cosine similarity for adjacent utterances in tfidf matrices..."
                tf_test = count_vectorizer.transform(utts)
                windowed_count_matrix = convert_to_windowed(tf_test, window_size)
                windowed_count_matrix = convert_to_windowed(tf_test, window_size)
                test_doc_cosines = make_cosine_list(windowed_count_matrix)
            test_vectors[doc_name] = test_doc_cosines
            doc_count += 1
    else: # switch == doc2vec
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
            test_vectors[test_doc] = test_doc_cosines
            test_index += 1

    print "Done.\n"

    print "\nScoring cosine curve depths, predicting boundaries..."
    test_index = 1
    test_total = len(test_vectors)
    for doc_name, cosines in test_vectors.iteritems():
        print "Test doc %i of %i: \t %s" % (test_index, test_total, doc_name)
        depth_scores, score_count, zero_count = score_depth(cosines, verbose=False)
        non_zero_depth_scores = [x for x in depth_scores if x != 0.0]
        predicted_boundaries, boundary_count = [], 0
        if seg_method == 'threshold':
            threshold = (np.mean(non_zero_depth_scores) - np.std(non_zero_depth_scores)) / 2
            for depth_score in depth_scores:
                if depth_score > threshold and depth_score != 0.0:
                    predicted_boundaries.append("BREAK")
                    boundary_count += 1
                else:
                    predicted_boundaries.append(None)
        elif type(seg_method) is int:
            threshold = (np.mean(non_zero_depth_scores) - np.std(non_zero_depth_scores)) / 2
            for depth_score in depth_scores:
                if depth_score > threshold and depth_score != 0.0:
                    predicted_boundaries.append("BREAK")
                    boundary_count += 1
                else:
                    predicted_boundaries.append(None)
        # There is one more boundary than there are depth scores
        predicted_boundaries.append(None)
        reverse_parse(doc_name, predicted_boundaries)
        test_index += 1
        # print "%i boundaries predicted, %i gold boundaries." % (boundary_count, gold_count)

        if mode == 'evaluate':
            evaluation = windowdiff(predicted_boundaries, gold_boundaries_set[doc_name], 3, boundary="BREAK",
                                    weighted=False)
            print "Window Diff Score:\t %f\n" % evaluation
            evaluations[doc_name] = evaluation
            with open(log_dir + timestamp, 'a') as log:
                log.write(doc_name + "\t"*3 + str(evaluation))

        elif mode == 'resolve':
            print "Resolving topic timings..."
            utt_start_times = get_utt_timing(doc_name)
            topic_start_times = [utt_start_times[0]]  # Initialise with start time of first utterance
            for index in xrange(len(predicted_boundaries)):
                if predicted_boundaries[index] is not None:
                    topic_start_times.append(utt_start_times[index])
            prg_name = doc_name.replace("_parsed","")
            prg_name = prg_name.replace(".txt",".wav")
            topic_timings[prg_name] = topic_start_times
    if mode == 'resolve' and play:
        import pickle
        pickle.dump(topic_timings, open("topic_timings", "wb"))
    print "-" * 30 + "  End  " + "-" * 30
