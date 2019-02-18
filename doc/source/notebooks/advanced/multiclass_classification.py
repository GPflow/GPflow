# Function to plot the predictions of the trained
# multi-class classification model in multiclass_classification.ipynb

import matplotlib.pyplot as plt
import numpy as np


colors = ['#1f77b4', '#ff7f0e', '#2ca02c']


def plot_posterior_predictions(m):

    f = plt.figure(figsize=(12,6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

    xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1,1)
    mu, var = m.predict_f(xx)
    mu, var = mu.copy(), var.copy()
    p, _ = m.predict_y(xx)
    
    a3.set_xticks([])
    a3.set_yticks([])
    
    for c in range(m.likelihood.num_classes):
        x = m.X.read_value()[m.Y.read_value().flatten()==c]
        
        color=colors[c]
        a3.plot(x, x*0, '.', color=color)
        a1.plot(xx, mu[:,c], color=color, lw=2, label='%d'%c)
        a1.plot(xx, mu[:,c] + 2*np.sqrt(var[:,c]), '--', color=color)
        a1.plot(xx, mu[:,c] - 2*np.sqrt(var[:,c]), '--', color=color)
        a2.plot(xx, p[:,c], '-', color=color, lw=2)
    
    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])
    
    a3.set_title('inputs X')
    a2.set_title('predicted mean label value \
                 $\mathbb{E}_{q(\mathbf{u})}[y^*|x^*, Z, \mathbf{u}]$')
    a1.set_title('posterior process \
                $\int d\mathbf{u} q(\mathbf{u})p(f^*|\mathbf{u}, Z, x^*)$')
    
    handles, labels = a1.get_legend_handles_labels()
    a1.legend(handles, labels)
    f.tight_layout()
    plt.show()


def plot_from_samples(m, samples, burn, thin):
    
    f = plt.figure(figsize=(12,6))
    a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
    a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
    a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])
    
    xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1,1)
    
    Fpred, Ypred = [], []

    for i, s in samples[burn::thin].iterrows():  # also possible as arguments to hmc.sample()
        feed_dict = m.sample_feed_dict(s)
        Ypred.append(m.predict_y(xx, feed_dict=feed_dict)[0])
        Fpred.append(m.predict_f_samples(xx, 1, feed_dict=feed_dict).squeeze())

    for i in range(m.likelihood.num_classes):
        x = m.X.read_value()[m.Y.read_value().flatten() == i]
        points, = a3.plot(x, x*0, '.')
        color = points.get_color()
        for F in Fpred:
            a1.plot(xx, F[:,i], color=color, lw=0.2, alpha=1.0)
        for Y in Ypred:
            a2.plot(xx, Y[:,i], color=color, lw=0.5, alpha=1.0)
    
    a2.set_ylim(-0.1, 1.1)
    a2.set_yticks([0, 1])
    a2.set_xticks([])  
    
    a3.set_xticks([])
    a3.set_yticks([])
    
    a3.set_title('inputs X')
    a2.set_title('predicted mean label value \
                 $\mathbb{E}_{q(\mathbf{u})}[y^*|x^*, Z, \mathbf{u}]$')
    a1.set_title('posterior process samples \
                $\int d\mathbf{u} q(\mathbf{u})p(f^*|\mathbf{u}, Z, x^*)$')
    