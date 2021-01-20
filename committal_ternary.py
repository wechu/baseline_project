import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
# import ipdb
sns.set_context('poster')
np.random.seed(3)

def softmax(z):
    zm = np.exp(z-z.max())
    return zm/zm.sum()

def entropy(vec):
    assert np.allclose(vec.sum(), 1)
    return (-vec*np.log(vec)).sum()

class Bandit:

    """Bandit class to generate the bandit problem and the pull arm function"""

    def __init__(self, K, means=None, std=None):
        """TODO: to be defined1. """
        self.K = K
        self.means = np.random.randn(K,)*5 if means is None else means
        self.std = np.ones(K,)*3 if std is None else std

    def pull(self, a):
        """Pull arm k out of K and return observed reward
        :k: arm to pull
        :returns: observed reward
        """
        r =  np.random.randn()*self.std[a]+self.means[a]
        # r =  (np.random.randn()*self.var[a]+1)*self.means[a]
        #self.update_arms()
        return r

    def reinitialize(self):
        self.means = np.random.randn(K,)*5
        self.std = np.ones(K,)*3


class Strategy(object):

    """Virtual class for all strategies (ubc, eps-greedy, policy gradient..)"""

    def run_round(self, n_round=5, episodes=1000, plot=False):
        # we compute mean and variance recursively
        M_reward = np.zeros(episodes,) #n x mean
        S_reward = np.zeros(episodes,) #n x var
        var_table = np.zeros((episodes, n_round, 4))
        grad_norm_table = np.zeros((episodes, n_round))
        entropy_table = np.zeros((episodes, n_round, 2))
        proba_table = np.zeros((n_round, episodes, 3))
        for i_round in range(n_round):
            self.reinitialize()
            #self.bandit.reinitialize()
            rewards = np.zeros(episodes,)
            for i in range(episodes):
                # Select action
                a = self.action()
                # Get reward for that action
                r = self.bandit.pull(a)
                #r = 100*r/self.bandit.means.max()
                # Update strategy with given action/reward pair
                self.observe(a, r)
                rewards[i] = r
                proba_table[i_round, i, :] = self.proba
        return proba_table

    def plot(self, meanR, stdR):
        best_reward = bandit.means.max()
        vec = np.arange(episodes)
        meanR, stdR = 100*meanR/best_reward, 50*stdR/best_reward
        # Plot the average curve and fill 1 sigma around
        plt.plot(vec, meanR, label=self.label())
        plt.fill_between(vec, meanR-stdR, meanR+stdR, alpha = 0.25)


class Policy_gradient(Strategy):

    """Policy based strategy"""

    def __init__(self, bandit, alpha=0.1):
      """Init for policy based strat.
      :alpha: learning rate
      """
      self.alpha = alpha
      self.bandit = bandit
      self.H = np.ones(self.bandit.K,)*0
      self.proba = np.ones_like(self.H,)/self.H.size
      self.r_m = 0
      self.counter = 0
      self.is_mu = False
      self.mu = np.ones_like(self.H,)/self.H.size

    def action(self):
      self.compute_baseline()
      if not self.is_mu:
        # On policy
        self.proba = softmax(self.H)
        a = np.argmax(np.random.multinomial(1, self.proba))
      else:
        # Off policy
        self.compute_is_mu()
        a = np.argmax(np.random.multinomial(1, self.mu))
      return a

    def observe(self, a, r):
      # self.r_m = self.counter/(1+self.counter)*self.r_m + 1/(1+self.counter)*r
      # TODO: add weight corrections for IS
      self.counter += 1
      #self.H -= self.alpha*(r - self.r_m)*self.proba
      #self.H[a] += self.alpha*(r - self.r_m)
      self.H[a]  += self.alpha*(r - self.r_m)/self.proba[a]

    def reinitialize(self):
      self.H *= 0
      self.proba = np.ones_like(self.H,)/self.H.size
      self.mu = np.ones_like(self.H,)/self.H.size
      self.counter = 0
      self.r_m = 0

    def set_H(self, H):
      self.H = H
      self.proba = softmax(self.H)
      
    def compute_baseline(self, typ='optimal', perturb=0):
      if typ == 'optimal':
        self.r_m = self.optimal_baseline()+perturb
      elif typ == 'mean':
        self.r_m = (self.proba*self.bandit.means).sum()
      else:
        raise NotImplementedError

    def compute_is_mu(self, b):
      self.proba = softmax(self.H)
      mu = self.proba*np.abs(self.bandit.means-b)*np.sqrt(1+(self.proba**2).sum()-2*self.proba)
     
      # mu = mu*0 + 1
      mu /= mu.sum()
      self.mu = mu

    def label(self):
      return r'Policy gradient ($\alpha = {}$)'.format(self.alpha)
      
    def variance_grad(self, b=None, mu=None, perturb=0):
      # self.r_m = 0
      self.compute_baseline(typ='optimal')
      if b is None:
        b = self.r_m+perturb
      
      s2 = (self.proba*(self.bandit.means - b)**2*(1+(self.proba**2).sum()-2*self.proba))
      # var_means = self.proba*(self.bandit.means-(self.bandit.means*self.proba).sum())
      # var_means = (var_means**2).sum()
      var_means = self.grad_norm()**2

      if mu is not None:
        #self.compute_is_mu(b)
        self.mu = mu
        ratio = self.proba/self.mu
        s2 = (self.mu*ratio**2*(self.bandit.means - b)**2*(1+(self.proba**2).sum()-2*self.proba))
        # s2 = ratio * s2

      var = s2.sum()-var_means
      # assert np.all(var >= 0), print(var)
      var = np.abs(var)
      return var

    def variances(self, b=None, mu=None, perturb=0):
      vanilla_var = self.variance_grad(b=0, perturb=perturb)
      baseline_var = self.variance_grad(b=b, perturb=perturb)
      mu_var = self.variance_grad(b=0, mu=mu, perturb=perturb)
      both_var = self.variance_grad(b=b, mu=mu, perturb=perturb)
      return np.array([vanilla_var, baseline_var, mu_var, both_var])

    def grad_norm(self):
      # vec = (1-2*self.proba)*self.bandit.means + self.bandit.means.sum()*self.proba
      # return np.linalg.norm(vec)
      var_means = self.proba*(self.bandit.means-(self.bandit.means*self.proba).sum())
      var_means = (var_means**2).sum()
      return np.sqrt(var_means)

    # def SNRs(self):
    #   grad_norm = self.grad_norm()
    #   variances = self.variances()
    #   return np.sqrt(grad_norm**2/(variances + grad_norm**2))

    def optimal_baseline(self):
      b = (self.proba*(self.bandit.means)*(1+(self.proba**2).sum()-2*self.proba)).sum()
      b /= (self.proba*(1+(self.proba**2).sum()-2*self.proba)).sum()
      return b


episodes = 5000
n_round = 5
#means = np.array([1, 0.8, -1])*10
means = np.array([6, 1, -2])
K = means.size
bandit = Bandit(K, means=means, std=means*0)

pg = Policy_gradient(bandit)

def var_p(p, pg=pg):
  H = np.log(p)
  pg.set_H(H)
  gamma=.5
  mu = (1-gamma)*pg.proba+gamma/3
  mu /= mu.sum()

  var = pg.variances(b=None, mu=mu)
  # grad_norm = pg.grad_norm()*np.ones_like(var)
  # snr = np.sqrt(grad_norm**2 / (grad_norm**2 + var))
  # if np.any(snr > 1):
  #   print('----')
  #   print(pg.H)
  #   print(grad_norm, var)
  #   print(snr)
  # if np.max(p) == 1:
  #     return 0
  return np.log(var[0]/var[2])
  #return var[1]

probs = pg.run_round()

scale = 30
fontsize = 18
figure, tax = ternary.figure(scale=scale)

points = [tuple(probs[0, i]) for i in range(probs.shape[1])]
tax.plot_colored_trajectory(points, linewidth=2.0)

tax.boundary(linewidth=2.0)
#tax.set_title(r"$\log_{10}\big( \frac{Var[IS]}{Var[b^\ast]}\big)$"+'\n\n', fontsize=fontsize)
# tax.right_corner_label(rf"$r_1 = {means[0]}$", fontsize=fontsize)
# tax.top_corner_label(rf"$r_2 = {means[1]}$", fontsize=fontsize)
# tax.left_corner_label(rf"$r_3 = {means[2]}$", fontsize=fontsize)
#tax.set_title("Shannon Entropy Heatmap")
tax.get_axes().axis('off')
tax.get_axes().set_aspect(2/np.sqrt(3))
tax.clear_matplotlib_ticks()
#cb = figure.colorbar('off')

tax.show()
