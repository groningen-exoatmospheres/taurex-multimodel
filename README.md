# taurex-multimodel
TauREx models with multiple regions:

describe multiple models in different parfiles and use the param fractions to separate the contributions.

Also add broadening fitting capabilities with adaptive model if needed:

keyword: wlbroadening_method = 'binned_convolution' or wlbroadening_method = 'binned_convolution_R'
param available: wlbroadening_default, wlbroadening_width, wlbroadening_width2

- binned_convolution uses sigma = wlbroadening_default + wlbroadening_width * lambda + wlbroadening_width2 * lambda**2
- binned_convolution_R uses sigma = 0.5 * lambda / (wlbroadening_default + wlbroadening_width * lambda + wlbroadening_width2 * lambda**2)
