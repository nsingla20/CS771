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
# f = 0.5
model=[]
def createFeatures( X ):
    # ln = X.shape[0]
    # return np.concatenate( (1-2*X,np.full((ln,1),1)),axis=1)
    return 1-2*X

def mux_inputs(Z_train):
    p1,p2 = np.packbits(Z_train[:,R:R+S],axis=-1),np.packbits(Z_train[:,R+S:R+2*S],axis=-1)
    p1,p2 = np.squeeze(p1), np.squeeze(p2)
    p1,p2 = np.right_shift(p1,8-S), np.right_shift(p2,8-S)
    p1,p2 = p1.astype(int), p2.astype(int)
    return p1,p2

def prepare(p1,p2,responses):
    ln = p1.shape[0]
    for i in range(ln):
        if p1[i]<p2[i]:
            responses[i] = bool(1-responses[i])
            tmp = p1[i]
            p1[i] = p2[i]
            p2[i] = tmp

def gog(ar,i,j,m):
    l = [False for _ in range(0,i-j+1)]
    l[0] = True
    for k in range(1,i-j+1):
        for jj in range(j,k+j):
            if k==i-j and jj == j:
                continue
            if (ar[k+j][jj] == m) and l[jj-j]:
                l[k] = True
                break
    return l[i-j]

def get_unique(records_array):
    idx_sort = np.argsort(records_array,axis=0)

    # sorts records array so all unique elements are together
    sorted_records_array = records_array[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True,axis=0)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])
    return vals,res

def find_path(ar,i,j):
    if i-j < 2:
        return -1
    ans = 0
    a = gog(ar,i,j,False)
    b = gog(ar,i,j,True)
    if a:
        if b:
            ans = 2
        else:
            ans = 0
    else:
        if b:
            ans = 1
        else:
            ans = -1
    return ans

def add_predict(challenges,i,j,model):
    l=[]
    ln = challenges.shape[0]
    arr = [[model[i][j].predict(challenges) if j<i else [0]*ln for j in range(N)] for i in range(N)]
    arr = np.array(arr)
    arr = arr.astype(bool)
    for k in range(ln):
        ar = arr[:,:,k]
        # ar = np.squeeze(ar)
        # print(ar.shape)
        ans = find_path(ar,i,j)
        # print(ans)
        if ans > -1 and ans < 2:
            l.append(bool(ans))
        else:
            l.append(ar[i][j])
    return l

def add_multi_predict(challenges,p1,p2,responses):
    challenges = challenges.astype(str)
    challenges = [''.join(cha) for cha in challenges]
    challenges = np.array(challenges)
    challenges_uq , idxs = get_unique(challenges)
    print(challenges.shape,challenges_uq.shape)
    nw_ch = []
    nw_p1 = []
    nw_p2 = []
    nw_rs = []
    ln = challenges_uq.shape[0]
    for k in range(ln):
        # print("go")
        l = idxs[k]
        ar = np.full((N,N),-1)
        for i in l:
            ar[p1[i]][p2[i]] = responses[i]
        for i in range(N):
            for j in range(i):
                if ar[i][j] == -1:
                    ans = find_path(ar,i,j)
                    if ans==2:
                        print(challenges_uq[k],i,j)
                    if ans > -1 and ans < 2:
                        ar[i][j] = bool(ans)
                        nw_ch.append(idxs[k][0])
                        nw_p1.append(i)
                        nw_p2.append(j)
                        nw_rs.append(bool(ans))
                        # print("added",i,j)
    return nw_ch,nw_p1,nw_p2,nw_rs





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
    # print(Z_train.shape)
    challenges = createFeatures(Z_train[:,:R])
    responses = Z_train[:,-1]
    p1,p2 = mux_inputs(Z_train)
    prepare(p1,p2,responses)

    # nw_ch,nw_p1,nw_p2,nw_rs = add_multi_predict(challenges,p1,p2,responses)

    # challenges = np.append(challenges,challenges[nw_ch],axis=0)
    # p1 = np.append(p1,nw_p1,axis=0)
    # p2 = np.append(p2,nw_p2,axis=0)
    # responses = np.append(responses,nw_rs,axis=0)

    ln = challenges.shape[0]
    model = [[LinearSVC() for j in range(i)]for i in range(N)]
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
                # ans = add_predict(challenges[l[i][j]],i,j,model)
                pred[l[i][j]] = ans
    pred[err] = 1- pred[err]
    # print(pred)
    return pred

if __name__ == "__main__":
    test_data = np.loadtxt("test.dat")
    train_data = np.loadtxt("train.dat")

    model=my_fit(train_data)

    pred = my_predict(test_data[:, :-1],model)

    print(np.average(test_data[:,-1] == pred))
