import pickle

#Loading model in

with open('SVM_model.pkl', rb) as file:
	model = pickle.load(file)
