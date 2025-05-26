# TauREx-MultiModel : A TauREx plugin to enable combination of 1D models in atmospheric retrievals.

A custom [TauREx](https://github.com/ucl-exoplanets/TauREx3_public) plugin for combining multiple 1D models in atmospheric retrievals. Supports different geometries: transit, eclipse, direct imaging. The multiple models are described in different parfiles (non described parts are inherited from the master model, allowing coupling of properties and free parameters). The contributions are combined according to their contribution fraction.

---
## 💾 Installation 

You can install TauREx-MultiModel using python in the plugin's folder:

```python
python3 setup.py install
```
---

## 📌 Purpose

This module computes the final spectrum from planets using multiple models. For instance the terminator can be modeled by multiple regions (e.g., one cloudy, one clear, one hazy). It also add instrument broadening fitting capabilities with the adaptive model mixin if needed:

keyword: wlbroadening_method = 'binned_convolution' or wlbroadening_method = 'binned_convolution_R'
param available: wlbroadening_default, wlbroadening_width, wlbroadening_width2

- binned_convolution uses sigma = wlbroadening_default + wlbroadening_width * lambda + wlbroadening_width2 * lambda**2
- binned_convolution_R uses sigma = 0.5 * lambda / (wlbroadening_default + wlbroadening_width * lambda + wlbroadening_width2 * lambda**2)

---
