from taurex.model import ForwardModel
from taurex.pressure import SimplePressureProfile
from .multichemistry import MultiChemistry
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
import numpy as np
from taurex.exceptions import InvalidModelException
from taurex.data.fittable import Fittable, derivedparam

class InvalidMultiModelException(InvalidModelException):
    """
    Exception that is called when the model is not physical
    """
    pass

class MultiTransitModel(ForwardModel):

    def __init__(self,
                 temperature_profiles = [None]*3,
                 chemistry = [None]*3,
                 pressure_min = [1e-6]*3,
                 pressure_max = [1e6]*3,
                 nlayers = [100]*3,
                 pressure_profile=[None]*3,
                 planet = None,
                 star = None,
                 observation = None,
                 contributions=[None]*3,
                 fractions = [None]*3,
                 use_cuda=False,):
        super().__init__('MultiTransitModel')

        self._use_cuda = use_cuda
        self._observation = observation
        self._fractions = fractions
        self.autofrac = True

        if not planet:
            planet = Planet()
        if not star:
            star = BlackbodyStar()

        self._active_chems = []
        self._inactive_chems = []
        self._mus = []
        self._temperatures = []
        self._pressures = []

        self.tpsClasses = []
        self.chemsClasses = []
        self.pressClasses = []
        self.contrClasses = []
        
        self._star = star
        self._planet = planet
        self._sub_models = self.create_models(temperature_profiles,
                           chemistry,
                           pressure_profile,
                           pressure_min,
                           pressure_max,
                           nlayers,
                           planet,
                           star,
                           contributions,)

    @property
    def chemistry(self):
        actives = np.array(self._active_chems)
        inactives = np.array(self._inactive_chems)
        mu = np.array(self._mus)
        chemistry = MultiChemistry(actives, inactives, mu, active_species=self.activeGases)
        return chemistry

    @property
    def phaseCurves(self):
        return self._saved_phase_model

    def defaultBinner(self):
        from taurex_phase.binning import MultiBinner
        wngrids = [self.nativeWavenumberGrid] * len(self._fractions)
        return MultiBinner(wngrids)
    
    def initialize_profiles(self):
        self.activeGases = []
        self._active_chems = []
        self._inactive_chems = []
        self._mus = []
        self._temperatures = []
        self._pressures = []
        for m in self._sub_models:
            m.initialize_profiles()
            for c in m.chemistry.activeGases:
                if c not in self.activeGases:
                    self.activeGases.append(c)
            self._active_chems.append(m.chemistry.activeGasMixProfile[...])
            self._inactive_chems.append(m.chemistry.inactiveGasMixProfile[...])
            self._mus.append(m.chemistry.muProfile[...])
            self._temperatures.append(m.temperatureProfile[...])
            self._pressures.append(m.pressureProfile[...])

    def change_fit_values(self, v, prefix):
        name, latex, fget, fset, mode, to_fit, bounds = v
        name = '{}_{}'.format(prefix, name)
        latex = '{}_{}'.format(prefix, latex)
        return name, latex, fget, fset, mode, to_fit, bounds

    #def create_phases(self, ngauss=8):
    #    raise NotImplementedError

    #def select_model(self):
    #    raise NotImplementedError

    #def check_exceptions(self):
    #    raise NotImplementedError

    def create_models(self, temperature_profile, chemistry, pressure_profile, pressure_min, pressure_max, nlayers, planet, star, contributions):

        self._planet = planet
        self._star = star
        sub_models = []

        for i in range(len(self._fractions)):
            pressure_min = None
            pressure_max = None
            m = self.create_a_model(temperature_profile[i], chemistry[i], pressure_profile[i], pressure_min, pressure_max, nlayers[i], planet, star, contributions[i])
            sub_models.append(m)
        return sub_models

    def create_a_model(self, temperature_profile, chemistry, pressure_profile, pressure_min, pressure_max, nlayers, planet, star, contributions):
        #day_tp = temperature_profile[0]
        #day_chem = chemistry[0]
        #day_contribs = contributions[0]
        theModel = self.select_model()

        #day_pressure_prof = pressure_profile[0]
        
        if pressure_profile is None:
            #day_pressure_min = pressure_min[0]
            #day_pressure_max = pressure_max[0]
            #day_nlayers = nlayers[0]
            day_pressure_prof = SimplePressureProfile(atm_min_pressure=pressure_min, atm_max_pressure=pressure_max,
                                                      nlayers=nlayers)

        transit = theModel(temperature_profile=temperature_profile, chemistry=chemistry, planet=planet, star=star,
                                     pressure_profile=pressure_profile)

        for c in contributions:
            transit.add_contribution(c)

        #self._phase_day = day_emission
        self.tpsClasses.append(transit.temperature)
        self.chemsClasses.append(transit.chemistry)
        self.pressClasses.append(transit.pressure)
        self.contrClasses.append(transit.contribution_list)
        return transit

    def generate_auto_factors(self, bounds=None):
        fraction_bounds = [0,1]
        num_frac = len(self._fractions)
        if self.autofrac:
            num_frac = num_frac-1
        for idx in range(num_frac):
            def read_frac(self, idx=idx):
                return self._fractions[idx]
            def write_frac(self, value, idx=idx):
                self._fractions[idx] = value
            self.add_fittable_param('fr_{}'.format(idx+1), '$fr_{}$'.format(idx+1), read_frac,
                                    write_frac, 'linear', False, fraction_bounds)

    def collect_fitting_parameters(self):
        self._fitting_parameters.update(self._planet.fitting_parameters())
        if self._star is not None:
            self._fitting_parameters.update(self._star.fitting_parameters())
        #if self._modelpipe is not None:
        #   self._fitting_parameters.update(self._modelpipe.fittingParameters)
        for m in self._sub_models:
            self._fitting_parameters.update(m.fittingParameters)
        self.debug('Available Fitting params: %s',
                   list(self._fitting_parameters.keys()))

    def select_model(self):
        from taurex.model import EmissionModel, TransmissionModel
        #from .transmissiontwo import TransmissionModelTwo
        tm = TransmissionModel

        if self._use_cuda:
            try:
                from taurex_cuda import EmissionCudaModel, TransmissionCudaModel
                tm = TransmissionCudaModel
                #em = EmissionCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning('Cuda plugin not found or not working')
        return tm

    @property
    def nLayers(self):
        return self._sub_models[0].nLayers

    @property
    def densityProfile(self):
        from taurex.constants import KBOLTZ
        # np.stack([self._phase_day,self._phase_term,self._phase_night])
        return np.stack(
            [m.densityProfile for m in self._sub_models])

    @property
    def altitudeProfile(self):
        from taurex.constants import KBOLTZ
        # np.stack([self._phase_day,self._phase_term,self._phase_night])
        return np.stack(
            [m.altitudeProfile for m in self._sub_models])

    @property
    def temperatureProfile(self):
        return np.array(self._temperatures)
       #return np.stack([t.profile for t in self._temperatures])

    @property
    def pressureProfile(self):
        return np.array(self._pressures)
        #return np.stack([p.profile for p in self._pressures])

    @property
    def scaleheight_profile(self):
        return np.stack([m.scaleheight_profile for m in self._sub_models])

    @property
    def gravity_profile(self):
        return np.stack([m.gravity_profile for m in self._sub_models])


    def determine_coupled_fitting(self, profiles):

        coupling = [list(map(lambda x: x is j, profiles)) for j in profiles]

        #fit_params = [{}, {}, {}, {}]
        fit_params = [{} for i in range(len(profiles)+1)]

        # Easy check 
        if all(coupling[0]):
            fit_params[-1].update(profiles[0].fitting_parameters())
        else:
            for idx, p in enumerate(coupling):
                if any(p[:idx]):
                    continue
                elif any(p[idx+1:]):
                    fit_params[-1].update(profiles[idx].fitting_parameters())
                else:
                    fit_params[idx].update(profiles[idx].fitting_parameters())

        return fit_params

    def determine_coupled_contributions(self, contribs):

        seen_contribution = []

        #fit_params = [{}, {}, {}, {}]
        fit_params = [{} for i in range(len(contribs)+1)]

        def is_in(obj, l):
            return any([obj is x for i, x in enumerate(l)])

        check_contribs = [list(range(i+1, len(contribs))) for i, c in
                          enumerate(contribs)]

        def check_all(obj, l, check):
            dup = False
            for c in check:
                dup |= is_in(obj, l[c])
            return dup

        for i, v in enumerate(zip(contribs, check_contribs)):
            contr, contrlist = v
            for c in contr:
                if is_in(c, seen_contribution):
                    continue
                else:
                    seen_contribution.append(c)
                if check_all(c, contribs, contrlist):
                    fit_params[-1].update(c.fitting_parameters())
                else:
                    fit_params[i].update(c.fitting_parameters())

        return fit_params

    @property
    def muProfile(self):
        return np.mean(np.array(self._mus),axis=0)

    @derivedparam(param_name='mu', param_latex='$\mu$', compute=True)
    def mu(self):
        from taurex.constants import AMU
        return self.muProfile[0]/AMU

    def collect_base_derived_params(self):
        self._derived_parameters = {}
        self._derived_parameters.update(self.derived_parameters())
        self._derived_parameters.update(self._planet.derived_parameters())
        self._derived_parameters.update(self._star.derived_parameters())
        #self._derived_parameters.update(self.chemistry.derived_parameters())


    def collect_base_fitting_params(self):
        self._fitting_parameters = {}
        self._fitting_parameters.update(self.fitting_parameters())
        self._fitting_parameters.update(self._planet.fitting_parameters())
        self._fitting_parameters.update(self._star.fitting_parameters())
        
        profiles = [self.tpsClasses,
                    self.chemsClasses,
                    self.pressClasses]
        
        for p in profiles:
            #m1_fit,m2_fit,m3_fit,coupled_fit = self.determine_coupled_fitting(p)
            m_fits = self.determine_coupled_fitting(p) 
            num_reg = len(m_fits)
            for i, m in enumerate(m_fits):
                if i == num_reg - 1:
                    for k,v in m.items():
                        self._fitting_parameters['{}'.format(k)] = v
                else:
                    for k,v in m.items():
                        self._fitting_parameters['m{}_{}'.format(i+1,k)] = self.change_fit_values(v, 'm'+str(i+1))

            # for k,v in m1_fit.items():
            #     self._fitting_parameters['m1_{}'.format(k)] = self.change_fit_values(v, 'm1')

            # for k,v in m2_fit.items():
            #     self._fitting_parameters['m2_{}'.format(k)] = self.change_fit_values(v, 'm2')

            # for k,v in m3_fit.items():
            #     self._fitting_parameters['m3_{}'.format(k)] = self.change_fit_values(v, 'm3')

            # for k,v in coupled_fit.items():
            #     self._fitting_parameters['{}'.format(k)] = v


        # Contributions
        contribs = self.contrClasses
        m_fits = self.determine_coupled_contributions(contribs)
        num_reg = len(m_fits)
        for i, m in enumerate(m_fits):
            if i == num_reg - 1:
                for k,v in m.items():
                    self._fitting_parameters['{}'.format(k)] = v
            else:
                for k,v in m.items():
                    self._fitting_parameters['m{}_{}'.format(i+1,k)] = self.change_fit_values(v, 'm'+str(i+1))
        #m1_fit,m2_fit,m3_fit,coupled_fit = self.determine_coupled_contributions(contribs)


        # for k,v in m1_fit.items():
        #     self._fitting_parameters['m1_{}'.format(k)] = self.change_fit_values(v, 'm1')

        # for k,v in m2_fit.items():
        #     self._fitting_parameters['m2_{}'.format(k)] = self.change_fit_values(v, 'm2')

        # for k,v in m3_fit.items():
        #     self._fitting_parameters['m3_{}'.format(k)] = self.change_fit_values(v, 'm3')

        # for k,v in coupled_fit.items():
        #     self._fitting_parameters['{}'.format(k)] = v

    def build(self):
        for m in self._sub_models:
            m.build()
        #self.collect_fitting_parameters()
        # NEEDS TO BE ENABLED
        self.initialize_profiles()
        self.generate_auto_factors()
        self.collect_base_fitting_params()
        self.collect_base_derived_params()

    @property
    def nativeWavenumberGrid(self):
        return self._sub_models[0].nativeWavenumberGrid

    def check_exceptions(self):
        if np.sum(self._fractions) > 1:
            raise InvalidModelException
        elif np.any(np.array(self._fractions) < 0):
            raise InvalidModelException

    def model(self, wngrid=None, cutoff_grid=True):
        from taurex.util.util import clip_native_to_wngrid

        #self.check_structure_exceptions()
        #self.check_exceptions()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)

        if self.autofrac:
            self._fractions[-1] = 1 - np.sum(self._fractions[:-1])
        self.check_exceptions()

        self.initialize_profiles()

        native_fluxes = []
        native_taus = []

        for idx, m in enumerate(self._sub_models):
            val = m.model(wngrid=native_grid,cutoff_grid=cutoff_grid)
            self._active_chems[idx] = m.chemistry.activeGasMixProfile[...]
            self._inactive_chems[idx] = m.chemistry.inactiveGasMixProfile[...]
            self._mus[idx] = m.chemistry.muProfile[...]
            self._temperatures[idx] = m.temperatureProfile[...]
            native_taus.append(val[-2])
            native_fluxes.append(val[1])

        final_flux = self.compute_final_flux(native_fluxes)
        
        return native_grid, final_flux, np.average(np.array(native_taus),axis=0), None

    def compute_final_flux(self, native_fluxes):
        fl = np.array(native_fluxes)
        we = np.array(self._fractions)
        return np.sum(fl*we[:,None], axis=0)

    def generate_profiles(self):
        from taurex.util.output import generate_profile_dict
        prof = generate_profile_dict(self)
        prof['mu_profile'] = self.chemistry.muProfile
        return prof

    def compute_error(self, samples, wngrid = None, binner=None):
        from taurex.util.math import OnlineVariance

        tp_profiles = OnlineVariance()
        active_variances = [OnlineVariance()]*len(self._sub_models)
        inactive_variances = [OnlineVariance()]*len(self._sub_models)

        #tau_profile = OnlineVariance()
        if binner is not None:
            binned_spectrum = OnlineVariance()
        else:
            binned_spectrum = None
        native_spectrum = OnlineVariance()

        for weight in samples(): #sample likelihood space and get their parameters
            res = self.model()
            #tau_profile.update(tau,weight=weight)
            native = res[1]
            tp_profiles.update(self.temperatureProfile,weight=weight)
            for i in range(len(self._sub_models)):
                active_variances[i].update(self.chemistry.activeGasMixProfile['region'+str(i)],weight=weight)
                inactive_variances[i].update(self.chemistry.inactiveGasMixProfile['region'+str(i)],weight=weight)
            native_spectrum.update(native,weight=weight)
            if binned_spectrum is not None:
                binned = binner.bin_model(res)[1]
                binned_spectrum.update(binned,weight=weight)

        profile_dict = {}
        spectrum_dict = {}

        tp_std = np.sqrt(tp_profiles.parallelVariance())
        profile_dict['temp_profile_std']=tp_std
        profile_dict['active_mix_profile_std']={} 
        profile_dict['inactive_mix_profile_std']={}

        active_std = []
        inactive_std = []
        for i in range(len(self._sub_models)):
            active_std.append(np.sqrt(active_variances[i].parallelVariance()))
            inactive_std.append(np.sqrt(inactive_variances[i].parallelVariance()))
            profile_dict['active_mix_profile_std']['m'+str(i+1)] = active_std[i]
            profile_dict['inactive_mix_profile_std']['m'+str(i+1)] = inactive_std[i]

        spectrum_dict['native_std'] = np.sqrt(native_spectrum.parallelVariance())
        if binned_spectrum is not None:
            spectrum_dict['binned_std'] = np.sqrt(binned_spectrum.parallelVariance())

        return profile_dict, spectrum_dict

    def write(self, output):

        self.model(self.nativeWavenumberGrid)
        model = super().write(output)

        self._planet.write(model)
        self._star.write(model)

        #model.write_array('phases',np.array(self._phases))
        for i in range(len(self._sub_models)):
            out = model.create_group('region+'+str(i))
            self._sub_models[i]._chemistry.write(out)
            self._sub_models[i]._temperature_profile.write(out)
            self._sub_models[i].pressure.write(out)
        return model