from taurex.model import ForwardModel

class BaseParameterTransitModel(ForwardModel):

    def __init__(self, name, planet=None,
                 star=None,
                 observation = None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 #west_fraction=None,
                 #east_fraction=None,
                 #west_par=None,
                 #east_par=None,
                 #poles_par=None,
                 parfiles = None,
                 use_cuda=False,):
        super().__init__(name)
        self._planet = planet
        self._star = star
        self._observation = observation
        self._default_pressure = pressure_profile
        if self._default_pressure is None:
            from taurex.pressure import SimplePressureProfile
            self._default_pressure = \
                SimplePressureProfile(nlayers,atm_min_pressure,atm_max_pressure)
        self._atm_min_pressure = atm_min_pressure
        self._atm_max_pressure = atm_max_pressure
        self._nlayers = nlayers
        self._default_temperature = temperature_profile
        self._default_chemistry = chemistry

        #self._west_par = west_par
        #self._east_par = east_par
        #self._poles_par = poles_par
        self._parfiles = parfiles
        self._use_cuda = use_cuda

    def defaultBinner(self):
        return self._multimodel.defaultBinner()

    def read_parameters(self, parfile):
        from taurex.parameter import ParameterParser
        from taurex.parameter.factory import generate_contributions

        pp = ParameterParser()
        if parfile is None:
            self.info('No parfile givenm using defaults')
            return self._default_temperature, self._default_chemistry, \
                self._default_pressure, self.contribution_list

        pp.read(parfile)

        tp = pp.generate_temperature_profile()
        tp = self._default_temperature if tp is None else tp

        chem = pp.generate_chemistry_profile()
        chem = self._default_chemistry if chem is None else chem

        press = pp.generate_pressure_profile()
        press = self._default_pressure if press is None else press

        try:
            contribs = generate_contributions(pp._raw_config['Model'])
        except KeyError:
            contribs = []

        contribs = self.contribution_list if len(contribs)==0 else contribs

        return tp, chem, press, contribs


    def setup_keywords(self):
        
        # Trying some fancy python shit
        #parfiles = [self._west_par, self._east_par, self._poles_par]
        parfiles = self._parfiles
        regions = list(map(self.read_parameters, parfiles))
        temp_profile, chemistry,\
             pressure, contribs = [[regions[i][j] for i in range(len(regions))] for j in range(len(regions[0]))]
        #r1, r2, r3 = map(self.read_parameters, parfiles)
        #temp_profile, chemistry,\
        #     pressure, contribs = map(list,zip(r1, r2, r3))
        self._nlayers = [p._nlayers for p in pressure]

        kwargs = dict(
                 temperature_profiles=temp_profile,
                 chemistry=chemistry,
                 pressure_profile=pressure,
                 planet=self._planet,
                 star=self._star,
                 observation = self._observation,
                 contributions=contribs,
                 pressure_min = self._atm_min_pressure,
                 pressure_max = self._atm_max_pressure,
                 nlayers = self._nlayers,
                 use_cuda=self._use_cuda,)

        return kwargs


    def initialize_profiles(self):
        self._multimodel.initialize_profiles()

    def create_model(self):
        raise NotImplementedError

    def build(self):
        self._multimodel = self.create_model()
        self._multimodel.build()
        self._fitting_parameters.update(self._multimodel.fittingParameters)

    @property
    def densityProfile(self):
        return self._multimodel.densityProfile

    @property
    def altitudeProfile(self):
        return self._multimodel.altitudeProfile

    @property
    def temperatureProfile(self):
        return self._multimodel.temperatureProfile

    @property
    def pressureProfile(self):
        return self._multimodel.pressureProfile

    @property
    def chemistry(self):
        return self._multimodel.chemistry

    @property
    def scaleheight_profile(self):
        return self._multimodel.scaleheight_profile

    @property
    def gravity_profile(self):
        return self._multimodel.gravity_profile
    
    def model(self, wngrid=None, cutoff_grid=True):
        """Computes the forward model for a wngrid"""
        return self._multimodel.model(wngrid=wngrid, cutoff_grid=cutoff_grid)
    
    def compute_error(self,  samples, wngrid=None, binner=None):
        return self._multimodel.compute_error(samples, wngrid=wngrid, binner=binner)

    def write(self, output):
        return self._multimodel.write(output)

    @property
    def nativeWavenumberGrid(self):
        return self._multimodel.nativeWavenumberGrid

class MultiParameterTransitModel(BaseParameterTransitModel):

    def __init__(self, planet=None,
                 star=None,
                 observation= None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 parfiles=None,
                 use_cuda=False,
                 fractions=None):
        
        super().__init__('MultiTransitParameter',planet=planet,
                         star=star,
                         observation = observation,
                         pressure_profile=pressure_profile,
                         temperature_profile=temperature_profile,
                         chemistry=chemistry,
                         nlayers=nlayers,
                         atm_min_pressure=atm_min_pressure,
                         atm_max_pressure=atm_max_pressure,
                         parfiles = parfiles,
                         use_cuda=use_cuda,
                         )

        self._fractions=fractions
    
    def create_model(self):
        from .multitransit import MultiTransitModel
        kwargs = self.setup_keywords()

        return MultiTransitModel(**kwargs,fractions=self._fractions)

    @classmethod
    def input_keywords(self):
        return ['multi_transit', ]

class MultiParameterEclipseModel(BaseParameterTransitModel):

    def __init__(self, planet=None,
                 star=None,
                 observation= None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 parfiles=None,
                 use_cuda=False,
                 fractions=None,
                 radius_scaling=False):
        
        super().__init__('MultiEclipseParameter',planet=planet,
                         star=star,
                         observation = observation,
                         pressure_profile=pressure_profile,
                         temperature_profile=temperature_profile,
                         chemistry=chemistry,
                         nlayers=nlayers,
                         atm_min_pressure=atm_min_pressure,
                         atm_max_pressure=atm_max_pressure,
                         parfiles = parfiles,
                         use_cuda=use_cuda,
                         )

        self._fractions=fractions
        self._radius_scaling=radius_scaling
    
    def create_model(self):
        from .multieclipse import MultiEclipseModel
        kwargs = self.setup_keywords()
        return MultiEclipseModel(**kwargs,fractions=self._fractions, radius_scaling=self._radius_scaling)

    @classmethod
    def input_keywords(self):
        return ['multi_eclipse', ]


class MultiParameterDirectImModel(BaseParameterTransitModel):

    def __init__(self, planet=None,
                 star=None,
                 observation= None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 parfiles=None,
                 use_cuda=False,
                 fractions=None,
                 radius_scaling=False):
        
        super().__init__('MultiDirectParameter',planet=planet,
                         star=star,
                         observation = observation,
                         pressure_profile=pressure_profile,
                         temperature_profile=temperature_profile,
                         chemistry=chemistry,
                         nlayers=nlayers,
                         atm_min_pressure=atm_min_pressure,
                         atm_max_pressure=atm_max_pressure,
                         parfiles = parfiles,
                         use_cuda=use_cuda,
                         )

        self._fractions=fractions
        self._radius_scaling = radius_scaling
    
    def create_model(self):
        from .multidirect import MultiDirectImModel
        kwargs = self.setup_keywords()

        return MultiDirectImModel(**kwargs,fractions=self._fractions, radius_scaling=self._radius_scaling)

    @classmethod
    def input_keywords(self):
        return ['multi_directimage', ]

