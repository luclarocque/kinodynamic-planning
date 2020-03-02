import numpy as np
import save_obj
from sklearn import svm
from sklearn.metrics import f1_score


#  returns a classifier trained using the key-value pairs in
#    training_dict.
def trainsvm(training_dict, c, Jth):
    X = []
    y = []
    for k in training_dict:
        p = k[0] + k[1]  # concatenates two states into one long tuple
        cost = training_dict[k][0]
        X.append(p)
        y.append(cost)

    y = np.array(y) < Jth
    classifier = svm.SVC(C=c, kernel='poly')
    # classifier = linear_model.LogisticRegression(C=1.0, solver='saga')

    # classifier = svm.SVR(kernel='sigmoid')
    # classifier = svm.SVR(kernel='poly', degree=3)

    print("Training...")
    classifier.fit(X, y)
    print("Training complete.")
    return classifier


def best_classifier(training_dict, Jth):
    COSTkeys = list(COST.keys())
    subdict = {k: COST[k] for k in COSTkeys[:100000]}
    cvdict = {k: COST[k] for k in COSTkeys[150000:200000]}
    bestf1 = 0.
    bestc = 0.05
    for c in np.linspace(0.05, 2.05, 5):
        classifier = trainsvm(subdict, c, Jth)
        i=0
        preds = []
        trues = []
        for k in cvdict:
            pred =  classifier.predict([k[0]+k[1]])
            trueval = cvdict[k][0] < Jth
            preds.append(pred)
            trues.append(trueval)
            if i<10:
                print("predicted: {}, true: {}".format(pred, trueval))
            i+=1
        f1 = f1_score(trues, preds)
        print("with C={} --> f1 = {}".format(c, f1))
        if f1 > bestf1:
            bestc = c
            bestf1 = f1
            best_class = classifier

    return best_class, bestc, bestf1


#  saves svm classifier trained on 'count' number of entries in COST, with Jth
def trainer(count, n, Jth):
    COST = save_obj.load_object('COST_full_{}.pkl'.format(n))
    COSTkeys = list(COST.keys())
    print("total number of entries in COST dict:", len(COSTkeys))
    subdict = {k: COST[k] for k in COSTkeys[:count]}

    classifier = trainsvm(subdict, 0.05, Jth)
    tot = 0
    ptot = 0
    wrongs = 0
    for k in COSTkeys[100000:200000]:
        truth = COST[k][0] < Jth
        pred = classifier.predict([k[0]+k[1]])
        if pred != truth: 
            wrongs += 1
            # print("Wrong.  ", k, ":", COST[k][0])
        tot += truth  # number of actual true values
        ptot += pred  # number of predicted true values
    print("Out of 100,000 experiments -- tot: {}, ptot: {}, wrongs: {}".format(tot, ptot, wrongs))
    print("percentage correct: {}".format(1-wrongs/100000.))

    save_obj.save_object(classifier, 'classifier{}node_J{}.pkl'.format(n, int(Jth)))


# prints the number of wrong classifications, and percent correct.
def tester(fname_classifier, Jth):
    COST = save_obj.load_object('COST_obs.pkl')
    classifier = save_obj.load_object(fname_classifier)
    COSTkeys = list(COST.keys())

    tot = 0
    ptot = 0
    wrongs = 0
    for k in COSTkeys[1000000:1200000]:
        truth = COST[k][0] < Jth
        pred = classifier.predict([k[0]+k[1]])
        if pred != truth: 
            wrongs += 1
        tot += truth
        ptot += pred
    print("tot: {}, ptot: {}, wrongs: {}".format(tot, ptot, wrongs))
    print("percentage correct: {}".format(1-wrongs/200000.))


if __name__ == "__main__":
    # trainer(150000, 200.)
    # tester('classifier60000_J150.pkl', 150.)

    # Jth = 145.
    # COST = save_obj.load_object('COST_full_500.pkl')

    trainer(50000, 2000, 150)

    # classifier, c, J = best_classifier(COST, Jth)
    # print("Best f1: {} indicates best C: {}".format(J, c))'

    






