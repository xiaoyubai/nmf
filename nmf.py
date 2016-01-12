import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.linalg import lstsq
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF as skNMF

class NMF(object):
    """
    Create a NMF class to that is initialized with a document matrix (bag of words or tf-idf) V.
    It also take parameters k (# of latent topics) and the maximum # of iterations to perform
    """
    def __init__(self, V, latent_topics, max_iter=20, error=0.01):
        """
        Initialize our weights (W) and features (H) matrices
        Initialize the weights matrix (W) with (positive) random values to be a n x k matrix, where n is the number of documents and k is the number of latent topics
        Initialize the feature matrix (H) to be k x m where m is the number of words in our vocabulary (i.e. length of bag).
        """
        self.V = V
        self.k = latent_topics
        self.max_iter = max_iter
        self.W = np.random.choice(1000, size=[self.V.shape[0], self.k])
        self.H = np.random.choice(1000, size=[self.k, self.V.shape[1]])
        self.error = error

    def update_H(self):
        """
        holding W fixed and minimizing the sum of squared errors predicting the document matrix
        """
        self.H = lstsq(self.W, self.V)[0]
        self.H[self.H<0] = 0

    def update_W(self):
        """
        Use the same lstsq solver to update W while holding H fixed
        """
        W_T = lstsq(self.H.T, self.V.T)[0]
        self.W = W_T.T
        self.W[self.W<0] = 0

    def fit(self):
        """
        Repeat update_H and update_W for a fixed number of iterations, or until convergence (i.e. change in cost(V, W*H) close to 0)
        """
        for i in xrange(self.max_iter):
            if self.cost() > self.error:
                print "itr %d | current cost: %f" % (i, self.cost())
                self.update_H()
                self.update_W()
            else:
                print "min cost met"

    def cost(self):
        """
        cost function is mean squared error
        """
        return mean_squared_error(self.V, np.dot(self.W, self.H))

    def key_feat_idx(self):
        np.argsort(self.H)


def load_data(file, X_col, y_col):
    """
    Load, tokenize and vectorize data
    """
    df = pd.read_pickle(file)
    X = df[X_col]
    y = df[y_col]
    vect = CountVectorizer(max_features=5000, stop_words='english')
    tok = vect.fit_transform(X)
    return df, X, y, tok, vect

def main():
    """
    Test function
    """
    print "---INITIALIZE VARIABLES AND LOAD DATA---"
    df, X, y, tok, vect = load_data('data/articles.pkl', 'content', 'section_name')
    feature_name = np.array(vect.get_feature_names())
    section_names = df['section_name'].unique()
    print "---RUN NMF---"
    nmf = NMF(tok.todense(), len(section_names))
    nmf.fit()
    print "---GET TOP FEATURES---"
    top_features = feature_name[np.argsort(nmf.H)]
    top_five_features = top_features[:, :-5:-1]
    print "---COMPARE OUR WONDERFUL, BEAUTIFUL FUNCTION WITH THEIR SHITTY IMPLEMENTATION---"
    tfdif = TfidfVectorizer(max_features=5000, stop_words='english')
    sk_tok = tfdif.fit_transform(X)

    sk_nmf = skNMF()
    sk_nmf_out = sk_nmf.fit_transform(sk_tok.todense())
    sk_mse = mean_squared_error(tok.todense(), sk_nmf_out)
    print "theirs | ours"
    print "%f | %f" %(sk_mse, nmf.cost())
    print "for your consideration regarding section topics"
    print top_five_features

if __name__ == '__main__':
    main()
