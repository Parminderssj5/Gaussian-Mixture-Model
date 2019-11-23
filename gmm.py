import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
import numpy as np
from scipy.stats import multivariate_normal

n_samples = 100
mu1, sigma1 = -4,1.2
mu2, sigma2 = 4,1.8
mu3, sigma3 = 0,1.6
x1 = np.random.normal(mu1, np.sqrt(sigma1),n_samples)
x2 = np.random.normal(mu2, np.sqrt(sigma2),n_samples)
x3 = np.random.normal(mu3, np.sqrt(sigma3),n_samples)
X = np.array(list(x1)+list(x2)+list(x3))
np.random.shuffle(X)


tmp=1
def pdf(data, mean: float, variance: float):
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  return s1 * s2

bins = np.linspace(np.min(X),np.max(X),100)
plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")
plt.plot(bins, pdf(bins, mu1, sigma1), color='red', label="True pdf")
plt.plot(bins, pdf(bins, mu2, sigma2), color='red')
plt.plot(bins, pdf(bins, mu3, sigma3), color='red')
plt.legend()
plt.plot()

k = 3
weights=np.ones((k)) / k
means=np.random.choice(X,k)
variances=np.random.random_sample(size=k)
X=np.array(X)

tmp=1
eps=1e-8
for step in range(101):
  if step % 5 == 0:
    plt.figure(figsize=(10,6))
    axes = plt.gca()
    plt.title("Iteration {}".format(step))
    plt.scatter(X, [0.005] * len(X), color='navy', s=30, marker=2, label="Train data")
    plt.plot(bins, pdf(bins, mu1, sigma1), color='grey', label="True pdf")
    plt.plot(bins, pdf(bins, mu2, sigma2), color='grey')
    plt.plot(bins, pdf(bins, mu3, sigma3), color='grey')
    plt.plot(bins, pdf(bins, means[0], variances[0]), color='blue', label="Cluster 1")
    plt.plot(bins, pdf(bins, means[1], variances[1]), color='green', label="Cluster 2")
    plt.plot(bins, pdf(bins, means[2], variances[2]), color='magenta', label="Cluster 3")
    plt.legend(loc='upper left')
    plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
    plt.show()
  likelihood = []
  for j in range(k):
    likelihood.append(pdf(X, means[j], np.sqrt(variances[j])))
  likelihood = np.array(likelihood)
  b = []
  for j in range(k):
    b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))
    means[j] = np.sum(b[j] * X) / (np.sum(b[j]+eps))
    variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j]+eps))
    weights[j] = np.mean(b[j])