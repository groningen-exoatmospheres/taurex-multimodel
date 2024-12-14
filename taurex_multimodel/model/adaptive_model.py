from taurex.mixin import ForwardModelMixin
import numpy as np
from taurex.data.fittable import fitparam, Fittable, derivedparam
from taurex.util.util import create_grid_res
from taurex.binning import FluxBinner

class AdaptiveForwardModel(ForwardModelMixin):
    def __init_mixin__(self, 
                 wlshift = [1.,0.0],
                 wlbroadening_method = 'box',
                 wlbroadening_width = 3,
                 wlbroadening_width2 = 0,
                 wlbroadening_default = 0,
                 max_wlbroadening = 0.1,
                 factor_cut = 5,
                 wlres = 15000,
                 ):
        self._wlshift = wlshift
        self._wlbroadening_method = wlbroadening_method
        self._wlbroadening_width = wlbroadening_width
        self._wlbroadening_width2 = wlbroadening_width2
        self._wlbroadening_default = wlbroadening_default
        self._max_wlbroadening = max_wlbroadening
        self._factor_cut = factor_cut
        self._wlres = wlres

        self.generate_waveshift_fitting_params()
        
    def gaussian(self, x, mean, std):
        return 1.0 / (np.sqrt(2.0 * np.pi) * std) * np.exp(-np.power((x - mean) / std, 2.0) / 2)

    def box_moving_average(self, O):
        windows = self._wlbroadening_width
        kernel = np.ones(windows) / windows
        return O[0], np.convolve(O[1], kernel, mode='same')
    
    def gaussian_moving_average(self, O):
        windows = self._wlbroadening_width
        x = np.linspace(-5*windows, 5*windows, windows)
        kernel = self.gaussian(x, 0, windows)
        kernel = kernel/np.sum(kernel)
        return O[0], np.convolve(O[1], kernel, mode='same')

    def low_res_convolved(self, O):
        native_grid, final_flux, taus, _ = O

        new_grid = create_grid_res(self._wlres, native_grid[0], native_grid[-1])
        fb = FluxBinner(new_grid[:,0], new_grid[:,1])

        Y = fb.bindown(native_grid, final_flux)
        X = 10000/Y[0], Y[1]

        X_conv = np.zeros(len(X[0]))
        for i in range(len(X[0])):
            std = np.clip(self._wlbroadening_default + self._wlbroadening_width*X[0][i] + self._wlbroadening_width2*X[0][i]**2, a_min=1e-20, a_max=self._max_wlbroadening)
            if i != len(X[0])-1:
                windows = np.maximum(1, int(self._factor_cut*std/np.abs(X[0][i+1]-X[0][i])))
            else:
                windows = np.maximum(1, int(self._factor_cut*std/np.abs(X[0][i]-X[0][i-1])))
            imin = np.maximum(0, i-windows)
            imax = np.minimum(len(X[0]), i+windows+1)
            x = X[0][imin:imax]
            y = self.gaussian(x, X[0][i], std)
            X_conv[imin:imax] += X[1][i]*y/np.sum(y)

        return 10000/X[0], X_conv
    
    def low_res_convolved_R(self, O):
        native_grid, final_flux, taus, _ = O

        new_grid = create_grid_res(self._wlres, native_grid[0], native_grid[-1])
        fb = FluxBinner(new_grid[:,0], new_grid[:,1])

        Y = fb.bindown(native_grid, final_flux)
        X = 10000/Y[0], Y[1]

        X_conv = np.zeros(len(X[0]))
        for i in range(len(X[0])):
            #std = np.clip(self._wlbroadening_default + self._wlbroadening_width*X[0][i] + self._wlbroadening_width2*X[0][i]**2, a_min=1e-20, a_max=self._max_wlbroadening)
            std = np.clip(0.5*X[0][i]/(self._wlbroadening_default + self._wlbroadening_width*X[0][i]+self._wlbroadening_width2*X[0][i]**2), a_min=1e-20, a_max=self._max_wlbroadening)
            if i != len(X[0])-1:
                windows = np.maximum(1, int(self._factor_cut*std/np.abs(X[0][i+1]-X[0][i])))
            else:
                windows = np.maximum(1, int(self._factor_cut*std/np.abs(X[0][i]-X[0][i-1])))
            imin = np.maximum(0, i-windows)
            imax = np.minimum(len(X[0]), i+windows+1)
            x = X[0][imin:imax]
            y = self.gaussian(x, X[0][i], std)
            X_conv[imin:imax] += X[1][i]*y/np.sum(y)

        return 10000/X[0], X_conv
        
    def model(self, wngrid=None, cutoff_grid=True):
        O = super().model(wngrid=wngrid, cutoff_grid=cutoff_grid)
        
        if self._wlbroadening_method == 'binned_convolution':
            adapt_grid, adapt_flux = self.low_res_convolved(O)
        elif self._wlbroadening_method == 'binned_convolution_R':
            adapt_grid, adapt_flux = self.low_res_convolved_R(O)
        elif self._wlbroadening_method == 'gaussian_box' and self._wlbroadening_width > 0:
            adapt_grid, adapt_flux = self.gaussian_moving_average(O)
        elif self._wlbroadening_method == 'box' and self._wlbroadening_width > 0:
            adapt_grid, adapt_flux = self.box_moving_average(O)
        else:
            adapt_grid = O[0]
            adapt_flux = O[1]
        
        wl = 10000/adapt_grid[::-1]
        adapt_grid = 10000/np.polyval(self._wlshift, wl)[::-1]

        return adapt_grid, adapt_flux, O[2], None

    #@fitparam(param_name='Wshift',
    #          param_latex='$Wshift$',
    #          default_fit=False,
    #          default_bounds=[-1e-5, 1e-5])
    #def waveShift(self):
    #    return self._wlshift

    #@waveShift.setter
    #def waveShift(self, value):
    #    self._wlshift = value

    def generate_waveshift_fitting_params(self):
        """Generates the fitting parameters
        """
        bounds = [-1,1]
        for idx, val in enumerate(self._wlshift):
            num = idx+1
            param_name = 'wlshift_{}'.format(num)
            param_latex = '$wlshift_{}$'.format(num)

            def read_wlshift(self, idx=idx):
                return self._wlshift[idx]

            def write_wlshift(self, value, idx=idx):
                self._wlshift[idx] = value

            fget_wl = read_wlshift
            fset_wl = write_wlshift
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_wl,
                                    fset_wl, 'linear', default_fit, bounds)

    @fitparam(param_name='Wbroad',
              param_latex='$Wbroad$',
              default_fit=False,
              default_bounds=[0.0, 0.0002])
    def waveBroad(self):
        return self._wlbroadening_width
    
    @waveBroad.setter
    def waveBroad(self, value):
        self._wlbroadening_width = value

    @fitparam(param_name='Wbroad2',
              param_latex='$Wbroad2$',
              default_fit=False,
              default_bounds=[0.0, 0.0002])
    def waveBroad2(self):
        return self._wlbroadening_width2
    
    @waveBroad2.setter
    def waveBroad2(self, value):
        self._wlbroadening_width2 = value
    
    @fitparam(param_name='Dbroad',
              param_latex='$Dbroad$',
              default_fit=False,
              default_bounds=[0.0, 0.0002])
    def defaultWaveBroad(self):
        return self._wlbroadening_default

    @defaultWaveBroad.setter
    def defaultWaveBroad(self, value):
        self._wlbroadening_default = value

    def write(self, output):
        model = super().write(output)
        model.write_scalar('Wbroad', self._wlbroadening_width)
        model.write_scalar('Dbroad', self._wlbroadening_default)
        model.write_scalar('Wshift', self._wlshift)
        model.write_string('Wbroad_method', self._wlbroadening_method)
        return model

    @classmethod
    def input_keywords(self):
        return ['adaptivemodel', 'adaptive']