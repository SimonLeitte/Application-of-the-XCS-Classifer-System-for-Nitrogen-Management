import datetime as dt
from pcse.base import SimulationObject, StatesTemplate, RatesTemplate, ParamTemplate
from pcse.traitlets import Float, Instance
from pcse import signals

class GrowingDegreeDayModelV2(SimulationObject):
    """A simple model to accumulate growing degree days up till GDDatMaturity. The model
    stores the maturity date and triggers a crop_finish signal when maturity is reached.    
    """

    class Parameters(ParamTemplate):
        BaseTemperature = Float
        GDDatMaturity = Float

    class StateVariables(StatesTemplate):
        GDD = Float
        DayofMaturity = Instance(dt.date)

    class RateVariables(RatesTemplate):
        rGDD = Float

    def initialize(self, day, kiosk, parameters):
        self.params = self.Parameters(parameters)
        self.states = self.StateVariables(kiosk, GDD=0.0, DayofMaturity=None)
        self.rates = self.RateVariables(kiosk)

    def calc_rates(self, day, drv):
        self.rates.rGDD = max(0, drv.TEMP - self.params.BaseTemperature)

    def integrate(self, day, delt):
        self.states.GDD += self.rates.rGDD * delt
        if self.states.GDD >= self.params.GDDatMaturity:
            self.states.DayofMaturity = day
            self._send_signal(day=day, signal=signals.crop_finish, crop_delete=True)



    def apply_fertilizer(self, day, amount_kg_ha: float, depth_cm: float = 5.0,
                         f_NH4N: float = 0.0, f_NO3N: float = 1.0,
                         f_orgmat: float = 0.0, cnratio: float = 0.0,
                         initial_age: float = 0.0):

        self._send_signal(
            signal=signals.apply_n_snomin,
            day=day,
            amount=amount_kg_ha,
            application_depth=depth_cm,
            f_NH4N=f_NH4N,
            f_NO3N=f_NO3N,
            f_orgmat=f_orgmat,
            cnratio=cnratio,
            initial_age=initial_age
        )
        print(f"[inject] Applied {amount_kg_ha} kg N/ha "
              f"(NH4={f_NH4N}, NO3={f_NO3N}, org={f_orgmat}) at depth={depth_cm} cm on {day}")
