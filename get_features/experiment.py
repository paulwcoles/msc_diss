from topic_tiling3 import *


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
# Doc2Vec
train_iter = 10
min_count = 1
doc2vec_window = 8
doc2vec_size = 100
sample = 1e-4
negative = 5
# TopicTiling
window_size = 3
# Quick parameters
n_features = 100
n_topics = 30
n_top_words = 20
n_iterations = 5
train_iter = 2

# OPERATION MODES
switch = argv[1]
if switch not in ['lda', 'doc2vec', 'tfidf']:
    print "Invalid mode."
    exit(1)

mode = 'evaluate'
# mode = 'resolve'
play = False

if __name__ == "__main__":
    timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
    training_dir = "../data/today/split/train/"
    # training_dir = "../data/today/split/train_small/"
    # test_dir = "../data/today/split/test/"
    test_dir = "../data/today/split/temp/"
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
        n_tiling_iterations = 2
        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=n_iterations,
                                           learning_method='online', learning_offset=50.,
                                           random_state=0, doc_topic_prior=doc_topic_prior,
                                           topic_word_prior=topic_word_prior)
        lda_train_test(n_tiling_iterations, tf_matrix)
    print "-" * 30 + "  End  " + "-" * 30