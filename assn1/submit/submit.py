import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
R,S=64,4
N = 2**S

def createFeatures( X ):
    return 1-2*X

def mux_inputs(Z_train):
    p1,p2 = np.packbits(Z_train[:,R:R+S],axis=-1),np.packbits(Z_train[:,R+S:R+2*S],axis=-1)
    p1,p2 = np.squeeze(p1), np.squeeze(p2)
    p1,p2 = np.right_shift(p1,8-S), np.right_shift(p2,8-S)
    p1,p2 = p1.astype(int), p2.astype(int)
    return p1,p2

def prepare(p1,p2):
    ln = p1.shape[0]
    err = []
    for i in range(ln):
        if p1[i]<p2[i]:
            tmp = p1[i]
            p1[i] = p2[i]
            p2[i] = tmp
            err.append(i)
    return err

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to train your model using training CRPs
    # The first 64 columns contain the config bits
    # The next 4 columns contain the select bits for the first mux
    # The next 4 columns contain the select bits for the second mux
    # The first 64 + 4 + 4 = 72 columns constitute the challenge
    # The last column contains the response
    Z_train = Z_train.astype(bool)
    challenges = createFeatures(Z_train[:,:R])
    responses = Z_train[:,-1]
    p1,p2 = mux_inputs(Z_train)
    err = prepare(p1,p2)
    responses[err] = 1 - responses[err]

    ln = challenges.shape[0]
    model = [[LogisticRegression(C = 5, tol = 0.001, penalty='l2') for j in range(i)]for i in range(N)]
    l =[[[]for j in range(i)] for i in range(N)]
    for i in range(ln):
        l[p1[i]][p2[i]].append(i)
    for i in range(N):
        for j in range(i):
            if l[i][j]:
                model[i][j].fit(challenges[l[i][j]],responses[l[i][j]])
            else:
                print(i)
    return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to make predictions on test challenges
    ln = X_tst.shape[0]
    X_tst = X_tst.astype(bool)
    challenges = createFeatures(X_tst[:,:R])
    p1,p2 = mux_inputs(X_tst)
    l =[[[]for j in range(i)] for i in range(N)]
    err = prepare(p1,p2)
    for i in range(ln):
        l[p1[i]][p2[i]].append(i)
    pred = np.empty(shape=(ln,))
    for i in range(N):
        for j in range(i):
            if l[i][j]:
                ans = model[i][j].predict(challenges[l[i][j]])
                pred[l[i][j]] = ans
    pred[err] = 1- pred[err]
    return pred

if __name__ == "__main__":
    test_data = np.loadtxt("test.dat")
    train_data = np.loadtxt("train.dat")

    model=my_fit(train_data)

    pred = my_predict(test_data[:, :-1],model)

    print(np.average(test_data[:,-1] == pred))
