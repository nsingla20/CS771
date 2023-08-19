import numpy as np
from sklearn.svm import LinearSVC

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
R = 64
S = 4

def pre_process(X):
	result = np.empty((len(X), (R+1)*15), dtype=X.dtype)
	new_arr = np.empty((15, 1), dtype=X.dtype)
	for i in range(len(X)):
		new_arr[0, 0]  = np.prod(X[i, [64, 65, 66, 67]]) - np.prod(X[i, [68, 69, 70, 71]])
		new_arr[1, 0]  = np.prod(X[i, [64, 65, 66]]) - np.prod(X[i, [68, 69, 70]])
		new_arr[2, 0]  = np.prod(X[i, [64, 65, 67]]) - np.prod(X[i, [68, 69, 71]])
		new_arr[3, 0]  = np.prod(X[i, [64, 66, 67]]) - np.prod(X[i, [68, 70, 71]])
		new_arr[4, 0]  = np.prod(X[i, [65, 66, 67]]) - np.prod(X[i, [69, 70, 71]])
		new_arr[5, 0]  = np.prod(X[i, [64, 65]]) - np.prod(X[i, [68, 69]])
		new_arr[6, 0]  = np.prod(X[i, [64, 66]]) - np.prod(X[i, [68, 70]])
		new_arr[7, 0]  = np.prod(X[i, [64, 67]]) - np.prod(X[i, [68, 71]])
		new_arr[8, 0]  = np.prod(X[i, [65, 66]]) - np.prod(X[i, [69, 70]])
		new_arr[9, 0]  = np.prod(X[i, [65, 67]]) - np.prod(X[i, [69, 71]])
		new_arr[10,0]  = np.prod(X[i, [66, 67]]) - np.prod(X[i, [70, 71]])
		new_arr[11,0]  = X[i, [64]] - X[i, [68]]
		new_arr[12,0]  = X[i, [65]] - X[i, [69]]
		new_arr[13,0]  = X[i, [66]] - X[i, [70]]
		new_arr[14,0]  = X[i, [67]] - X[i, [71]]
		result[i, :-15] = (X[i, :R]*new_arr).reshape((R*15,))
		result[i, -15:] = new_arr.T

	return np.hstack((X, result))

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
	model = LinearSVC(random_state=0, dual = False, C = 10000, max_iter=2500)
	model.fit(pre_process(Z_train[:, :-1]), Z_train[:, -1])
	return model					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges

	pred = model.predict(pre_process(X_tst))
	return pred

if __name__ == "__main__":
    test_data = np.loadtxt("train.dat")
    train_data = np.loadtxt("train.dat")

    model=my_fit(train_data)

    pred = my_predict(test_data[:, :-1],model)

    print(np.average(test_data[:,-1] == pred))
