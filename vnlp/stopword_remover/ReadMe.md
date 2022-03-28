#### StopWord Remover

- Static stopwords list are taken from https://github.com/ahmetax/trstop and some minor improvements are done by removing numbers from it

- Dynamic stopword algorithm is implemented according to two papers below:
    - Saif, Fernandez, He, Alani. 
    “On Stopwords, Filtering and Data Sparsity for Sentiment Analysis of Twitter”.  
    Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14), pp. 810–817, 2014.

    - Automatic cut-point of stop-words is determined according to:
    Satopaa, Albrecht, Irwin, Raghavan.
    Detecting Knee Points in System Behavior”.  
    Distributed Computing Systems Workshops (ICDCSW), 2011 31st International Conference, 2011.