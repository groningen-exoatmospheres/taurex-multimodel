import numpy as np
from taurex.data.fittable import derivedparam


class MultiChemistry(object):

    def __init__(self, actives, inactives, mu, active_species=[]):
        self._chem = [[actives[i],inactives[i]] for i in range(len(actives))]
        self._active_profiles = actives
        self._inactive_profiles = inactives
        self._active_species = active_species
        self._mus = mu

    @property
    def activeGases(self):
        return self._active_species

    @property
    def activeGasMixProfile(self):
        mix = {}
        for idx, c in enumerate(self._chem):
            mix['region'+str(idx)] = c[0]
        #mix['day'] = self._chem[0][0]
        #mix['term'] = self._chem[1][0]
        #mix['night'] = self._chem[2][0]
        return mix

    @property
    def inactiveGasMixProfile(self):
        mix = {}
        for idx, c in enumerate(self._chem):
            mix['region'+str(idx)] = c[1]
        return mix
    
    
    @property
    def muProfile(self):
        return np.mean(self._mus,axis=0)

    @derivedparam(param_name='mu', param_latex='$\mu$', compute=True)
    def mu(self):
        from taurex.constants import AMU
        return self.muProfile[0]/AMU

    @property
    def hasCondensates(self):
        return False
