from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from matplotlib import pyplot as plt

# matched sample data
p = 1000
mu = np.zeros(p)
sg = np.eye(p)
mu_diff = 2 * np.ones(10)
mu2 = np.concatenate((mu_diff, mu[10:]))

training_sizes = np.array([5e0, 5e1, 5e2], dtype=int)
ntest = int(1e5)
num_trials = 10

acc = {}
acc['matched'] = np.zeros((num_trials, len(training_sizes)))
acc['unmatched'] = np.zeros((num_trials, len(training_sizes)))

for i, ntrain in enumerate(training_sizes):
    print(f'##### ntrain = {ntrain} #####')
    for trial in range(num_trials):
        print(f'trial {trial}')
        Xtrain_0 = np.random.multivariate_normal(mean=mu, cov=sg, size=int(ntrain/2))
        Xtrain_1 = Xtrain_0.copy()
        Xtrain_1[:, :10] = np.random.multivariate_normal(mean=mu_diff, cov=sg[:10, :10], size=int(ntrain/2))
        Xtrain_matched = np.concatenate((Xtrain_0, Xtrain_1), axis=0)
        Xtrain_1 = np.random.multivariate_normal(mean=mu2, cov=sg, size=int(ntrain/2))
        Xtrain_unmatched = np.concatenate((Xtrain_0, Xtrain_1), axis=0)
        Ytrain = np.concatenate((np.zeros(int(ntrain/2), dtype=int), np.ones(int(ntrain/2), dtype=int)))
        Xtest_0 = np.random.multivariate_normal(mean=mu, cov=sg, size=int(ntest/2))
        Xtest_1 = Xtest_0.copy()
        Xtest_1[:, :10] = np.random.multivariate_normal(mean=mu_diff, cov=sg[:10, :10], size=int(ntest/2))
        Xtest_matched = np.concatenate((Xtest_0, Xtest_1), axis=0)
        Xtest_1 = np.random.multivariate_normal(mean=mu2, cov=sg, size=int(ntest/2))
        Xtest_unmatched = np.concatenate((Xtest_0, Xtest_1), axis=0)
        Ytest = np.concatenate((np.zeros(int(ntest/2), dtype=int), np.ones(int(ntest/2), dtype=int)))

        clf = LinearDiscriminantAnalysis()
        clf.fit(Xtrain_matched, Ytrain)
        acc['matched'][trial, i] = clf.score(Xtest_unmatched, Ytest)

        clf = LinearDiscriminantAnalysis()
        clf.fit(Xtrain_unmatched, Ytrain)
        acc['unmatched'][trial, i] = clf.score(Xtest_unmatched, Ytest)
print(acc)
