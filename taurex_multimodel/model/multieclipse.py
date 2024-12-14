import typing as t
import numpy as np
import numpy.typing as npt

from taurex.model import ForwardModel, EmissionModel
from taurex.pressure import PressureProfile, SimplePressureProfile
from taurex.planet import Planet
from taurex.stellar import Star, BlackbodyStar
from taurex.exceptions import InvalidModelException
from .multichemistry import MultiChemistry
from .multitransit import MultiTransitModel

from taurex.chemistry import Chemistry
from taurex.constants import PI
from taurex.core import derivedparam
#from taurex.log import setup_log
#from taurex.output import OutputGroup
from taurex.temperature import TemperatureProfile
from taurex.util.emission import black_body

if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object

class MultiEclipseModel(MultiTransitModel):

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
                 use_cuda=False,
                 radius_scaling=False):
        self._radius_scaling = radius_scaling
        
        super().__init__(temperature_profiles=temperature_profiles, 
                 chemistry = chemistry,
                 pressure_min = pressure_min,
                 pressure_max = pressure_max,
                 nlayers = nlayers,
                 pressure_profile=pressure_profile,
                 planet = planet,
                 star = star,
                 observation = observation,
                 contributions=contributions,
                 fractions = fractions,
                 use_cuda=use_cuda,)
    
    def select_model(self):
        #from taurex.model import EmissionModel, TransmissionModel
        #from .transmissiontwo import TransmissionModelTwo
        em = EmissionModel

        if self._use_cuda:
            try:
                from taurex_cuda import EmissionCudaModel
                em = EmissionCudaModel
                #em = EmissionCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning('Cuda plugin not found or not working')
        
        if self._radius_scaling:
            self.info('Switching to radius scale model')
            em = EmissionModelRadiusScale

        return em

class EmissionModelRadiusScale(EmissionModel):
    """A forward model for eclipses."""

    def __init__(
        self,
        planet = None,
        star = None,
        pressure_profile = None,
        temperature_profile = None,
        chemistry = None,
        nlayers = 100,
        atm_min_pressure = 1e-4,
        atm_max_pressure = 1e6,
        #contributions = None,
        ngauss = 4,
    ):
        super().__init__(
            planet = planet,
            star = star,
            pressure_profile = pressure_profile,
            temperature_profile = temperature_profile,
            chemistry = chemistry,
            nlayers = nlayers,
            atm_min_pressure = atm_min_pressure,
            atm_max_pressure = atm_max_pressure,
            #contributions = contributions,
            ngauss = ngauss,
        )


    def evaluate_emission_ktables(  # noqa: C901
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Evaluate emission flux on quadratures using ktables."""
        from taurex.contributions import AbsorptionContribution
        from taurex.util import compute_dz

        dz = compute_dz(self.altitudeProfile)
        total_layers = self.nLayers
        density = self.densityProfile
        wngrid_size = wngrid.shape[0]
        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))
        layer_tau = np.zeros(shape=(1, wngrid_size))
        tmp_tau = np.zeros(shape=(1, wngrid_size))
        dtau = np.zeros(shape=(1, wngrid_size))
        planet_radius = self._planet.fullRadius
        planet_layer_radius = self.altitudeProfile + self._planet.fullRadius

        mol_type = AbsorptionContribution

        non_molecule_absorption = [
            c for c in self.contribution_list if not isinstance(c, mol_type)
        ]
        contrib_types = [type(c) for c in self.contribution_list]
        molecule_absorption = None
        if mol_type in contrib_types:
            molecule_absorption = self.contribution_list[contrib_types.index(mol_type)]

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]

        # Do surface first
        # for layer in range(total_layers):
        for contrib in non_molecule_absorption:
            contrib.contribute(
                self, 0, total_layers, 0, 0, density, surface_tau, path_length=dz
            )

        surface_tau = surface_tau * _mu
        if molecule_absorption is not None:
            for idx, m in enumerate(_mu):
                tmp_tau[...] = 0.0
                molecule_absorption.contribute(
                    self, 0, total_layers, 0, 0, density, tmp_tau, path_length=dz * m
                )
                surface_tau[idx] += tmp_tau[0]

        self.debug("density = %s", density[0])
        self.debug("surface_tau = %s", surface_tau)

        planck_term = black_body(wngrid, temperature[0]) / PI

        intensity = planck_term * (np.exp(-surface_tau))

        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in non_molecule_absorption:
                contrib.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contrib.contribute(
                    self, layer, layer + 1, 0, 0, density, dtau, path_length=dz
                )

            k_dtau = None
            k_layer = None
            wg = None
            if molecule_absorption is not None:
                wg = molecule_absorption.weights
                ng = len(wg)
                sigma = molecule_absorption.sigma_xsec

                k_layer = contribute_ktau_emission(
                    layer + 1,
                    total_layers,
                    0,
                    sigma,
                    density,
                    dz,
                    wg,
                    wngrid_size,
                    0,
                    ng,
                )

                k_dtau = contribute_ktau_emission(
                    layer, layer + 1, 0, sigma, density, dz, wg, wngrid_size, 0, ng
                )
                k_dtau += k_layer

            dtau += layer_tau

            self.debug("dtau[%s]=%s", layer, dtau)
            planck_term = black_body(wngrid, temperature[layer]) / PI * (planet_layer_radius[layer]**2) / (planet_radius**2)
            self.debug("planck_term[%s]=%s,%s", layer, temperature[layer], planck_term)

            dtau_calc = np.exp(-dtau * _mu)
            layer_tau_calc = np.exp(-layer_tau * _mu)

            if molecule_absorption is not None:
                dtau_calc *= np.sum(np.exp(-k_dtau * _mu[:, None]) * wg, axis=-1)
                layer_tau_calc *= np.sum(np.exp(-k_layer * _mu[:, None]) * wg, axis=-1)

            intensity += planck_term * (layer_tau_calc - dtau_calc)

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau)

            if molecule_absorption is not None:
                if k_dtau.min() < self._clamp:
                    dtau_calc *= np.sum(np.exp(-k_dtau) * wg, axis=-1)
                if k_layer.min() < self._clamp:
                    layer_tau_calc *= np.sum(np.exp(-k_layer) * wg, axis=-1)

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau, float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]

        self.debug("intensity: %s", intensity)

        return intensity, _mu, _w, tau

    

    def evaluate_emission(
        self, wngrid: npt.NDArray[np.float64], return_contrib: bool
    ) -> t.Tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Evaluate emission flux on quadratures."""
        if self.usingKTables:
            return self.evaluate_emission_ktables(wngrid, return_contrib)

        dz = self.deltaz

        planet_radius = self._planet.fullRadius
        
        planet_layer_radius = self.altitudeProfile + self._planet.fullRadius

        total_layers = self.nLayers

        density = self.densityProfile

        wngrid_size = wngrid.shape[0]

        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))

        layer_tau = np.zeros(shape=(1, wngrid_size))

        dtau = np.zeros(shape=(1, wngrid_size))

        # Do surface first
        # for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(
                self, 0, total_layers, 0, 0, density, surface_tau, path_length=dz
            )
        self.debug("density = %s", density[0])
        self.debug("surface_tau = %s", surface_tau)

        planck_term = black_body(wngrid, temperature[0]) / PI

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        intensity = planck_term * (np.exp(-surface_tau * _mu))

        self.debug("I1_pre %s", intensity)
        # Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(
                    self,
                    layer + 1,
                    total_layers,
                    0,
                    0,
                    density,
                    layer_tau,
                    path_length=dz,
                )
                contrib.contribute(
                    self, layer, layer + 1, 0, 0, density, dtau, path_length=dz
                )
            # for contrib in self.contribution_list:

            self.debug("Layer_tau[%s]=%s", layer, layer_tau)

            dtau += layer_tau

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau)

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau, float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]

            self.debug("dtau[%s]=%s", layer, dtau)
            planck_term = black_body(wngrid, temperature[layer]) / PI * (planet_layer_radius[layer]**2) / (planet_radius**2)
            self.debug("planck_term[%s]=%s,%s", layer, temperature[layer], planck_term)

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau * _mu)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau * _mu)

            intensity += planck_term * (layer_tau_calc - dtau_calc)

        self.debug("intensity: %s", intensity)

        return intensity, _mu, _w, tau

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "emission_radscale",
            "eclipse_radscale",
        )