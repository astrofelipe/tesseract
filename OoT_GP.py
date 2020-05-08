import argparse
import juliet
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Out of transit GP fit')
parser.add_argument('File', type=str, help='Light curve file with transits (1 instrument)')
parser.add_argument('P', type=float, help='Period')
parser.add_argument('t0', type=float, help='t0')
parser.add_argument('dur', type=float, help='Duration')
#parser.add_argument('--mdilution', type=float, default=None)

args = parser.parse_args()
P   = args.P
t0  = args.t0
dur = args.dur

t,f,e = np.genfromtxt(args.File, usecols=(0,1,2), unpack=True)

phase = juliet.utils.get_phases(t, P, t0)
omask = np.abs(phase) > 1.5*dur
#print(len(omask), omask.sum())

time, flux, ferr = {}, {}, {}
time['inst'], flux['inst'], ferr['inst'] = t[omask], f[omask], e[omask]

params = ['mdilution_inst', 'mflux_inst', 'sigma_w_inst',
          'GP_sigma_inst', 'GP_timescale_inst', 'GP_rho_inst']
dists  = ['fixed', 'normal', 'loguniform', 'loguniform', 'loguniform', 'loguniform']
hyper  = [1, [0, 0.1], [0.1, 1e4], [1e-3, 1e3], [1e-6, 1e3], [1e-6, 1e6]]

priors = {}
for par, dis, hyp in zip(params, dists, hyper):
    priors[par] = {}
    priors[par]['distribution'], priors[par]['hyperparameters'] = dis, hyp

dataset = juliet.load(priors=priors, t_lc=time, y_lc=flux, yerr_lc=ferr, verbose=True,
                      GP_regressors_lc=time, out_folder='GPO_' + args.File.split('.')[0])

results = dataset.fit(n_live_points=500)#, use_dynesty=True, dynesty_nthreads=30)

model_fit = results.lc.evaluate('inst', t=t, GPregressors=t)
#gp_fit    = results.lc.model['inst']['GP']

fig, ax = plt.subplots(figsize=[15,6], nrows=2, sharex=True)

ax[0].errorbar(t, f, yerr=e, fmt='.', color='k')
ax[0].plot(t, model_fit, color='r', zorder=100)

ax[1].errorbar(t, (f-model_fit)*1e6, yerr=e*1e6, fmt='.', color='k')

plt.show()
