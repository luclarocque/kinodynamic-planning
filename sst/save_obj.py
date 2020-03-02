import cPickle as pickle

def save_object(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
        