import numpy as np
from sklearn.svm import LinearSVC
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
f = 0.5
model=[]
def createFeatures( X ):
    # ln = X.shape[0]
    # return np.concatenate( (1-2*X,np.full((ln,1),1)),axis=1)
    return 1-2*X
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y

def mux_inputs(Z_train):
    ln = Z_train.shape[0]
    p1,p2 = np.empty(shape=(ln,),dtype=int) , np.empty(shape=(ln,),dtype=int)
    # p1,p2 = tmp*(Z_train[:,R:(R+S)]) , tmp*(Z_train[:,(R+S):-1])
    for i in range(ln):
        p1[i] = bool2int(Z_train[i,R:(R+S)])
        p2[i] = bool2int(Z_train[i,(R+S):(R+2*S)])
        # print(p1[i],p2[i])
    return p1,p2

def find_path(ar,r):
    l = [0 for j in range(0,r)]
    l[0]=0
    for k in range(1,r):
        for j in range(k):
            if ar[k][j]:
                l[k] += l[j] + 1
            else:
                l[k] += l[j] - 1
    return l[r-1]


def multi_predict(model,challenge,i,j):
    ar = [[model[ii][jj].predict(challenge)[0] for jj in range(j,ii)] for ii in range(j,i+1)]
    # print(ar)
    r=i-j+1
    ans = find_path(ar,r)
    return np.sign(ans)


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
    ln = Z_train.shape[0]
    # print(Z_train.shape)
    challenges = createFeatures(Z_train[:,:R])
    responses = Z_train[:,-1]
    p1,p2 = mux_inputs(Z_train)
    model = [[LogisticRegression() for j in range(N)]for i in range(N)]
    l =[[[]for j in range(i)] for i in range(N)]
    for i in range(ln):
        if p1[i]-p2[i]<0:
            responses[i] = 1-responses[i]
        l[max(p1[i],p2[i])][min(p1[i],p2[i])].append(i)
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
    l =[[[]for j in range(N)] for i in range(N)]
    err = []
    for i in range(ln):
        if p1[i]-p2[i]<0:
            err.append(i)
        l[max(p1[i],p2[i])][min(p1[i],p2[i])].append(i)
    pred = np.empty(shape=(ln,))
    for i in range(N):
        for j in range(i):
            if l[i][j]:
                ans = model[i][j].predict(challenges[l[i][j]])
                # ans = multi_predict(model,challenges[l[i][j]],i,j)
                pred[l[i][j]] = ans
    pred[err] = 1- pred[err]
    # print(pred)
    return pred

if __name__ == "__main__":
    test_data = np.loadtxt("test.dat")
    train_data = np.loadtxt("train.dat")

    model=my_fit(test_data)

    pred = my_predict(test_data[:, :-1],model)

    print(np.average(test_data[:,-1] == pred))
