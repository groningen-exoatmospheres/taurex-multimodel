
import typing as t
import numpy as np
import numpy.typing as npt

from taurex.model import ForwardModel, EmissionModel, DirectImageModel
from taurex.pressure import PressureProfile, SimplePressureProfile
from taurex.planet import Planet
from taurex.stellar import Star, BlackbodyStar
from taurex.exceptions import InvalidModelException
from .multichemistry import MultiChemistry
from .multitransit import MultiTransitModel
from .multieclipse import EmissionModelRadiusScale

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

class MultiDirectImModel(MultiTransitModel):

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
        #from taurex.model import EmissionModel, TransmissionModel, DirectImageModel
        #from .transmissiontwo import TransmissionModelTwo
        di = DirectImageModel

        if self._use_cuda:
            try:
                from taurex_cuda import DirectImageCudaModel
                di = DirectImageCudaModel
                #em = EmissionCudaModel
            except (ModuleNotFoundError, ImportError):
                self.warning('Cuda plugin not found or not working')

        if self._radius_scaling:
            self.info('Switching to radius scale model')
            di = DirectImageRadiusScaleModel

        return di

class DirectImageRadiusScaleModel(EmissionModelRadiusScale):
    """A forward model for direct image."""

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

    def compute_final_flux(
        self, f_total: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the final flux.

        This is the emission flux that is observed at the telescope directly
        from an exo-planet.

        """
        star_distance_meters = self._star.distance * 3.08567758e16
        planet_radius = self._planet.fullRadius

        return (
            (f_total * (planet_radius**2))
            / (star_distance_meters**2)
        ) 

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "direct_radscale",
            "directimage_radscale",
        )