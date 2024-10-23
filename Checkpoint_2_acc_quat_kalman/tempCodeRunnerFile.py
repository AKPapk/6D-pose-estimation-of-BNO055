ov[2,2]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[2] + 2*np.sqrt(co