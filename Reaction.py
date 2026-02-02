from DVar import DVar
from thermo import ChemicalConstantsPackage, HeatCapacityGas, HeatCapacityLiquid
import warnings
from math import e as exp

class Reaction:
    def get_thermo_constants(self):
        self.thermodynamic_constants = {}
        self.Cp_gases_objs = {}
        self.Cp_liq_objs = {}
        for component in self.component_names:
            self.thermodynamic_constants[component] = ChemicalConstantsPackage.constants_from_IDs([component])
            self.Cp_gases_objs[component] = HeatCapacityGas(CASRN=self.thermodynamic_constants[component].CASs[0])
            self.Cp_liq_objs[component] = HeatCapacityLiquid(CASRN=self.thermodynamic_constants[component].CASs[0])

    def calc_reaction_HSG(self):
        delta_H_formation = []
        delta_H_Temp_change = []
        # calculate S too to give the user the reaction equilibrium constant
        # K = exp(-deltaG/RT), dG = dH - T dS
        # so deltaG = deltaH - T deltaS
        delta_S_formation = []
        delta_S_Temp_change = []

        rx_component_names = self.component_names
        rx_base_component = self.base_component
        rx_stoic = self.stoic

        for component in self.component_names:
            if self.liquid_phase:
                raise ValueError("Liquid phase calculation not yet implemented.")
            if not self.liquid_phase:
                delta_H_formation.append(rx_stoic[component]*self.thermodynamic_constants[component].Hfgs[0])
                int_Cp_dT = self.Cp_gases_objs[component].T_dependent_property_integral(298, self.T.val)
                delta_H_Temp_change.append(rx_stoic[component]*int_Cp_dT)

                delta_S_formation.append(rx_stoic[component]*self.thermodynamic_constants[component].Sfgs[0])
                int_Cp_over_T_dT = self.Cp_gases_objs[component].T_dependent_property_integral_over_T(298, self.T.val)
                delta_S_Temp_change.append(rx_stoic[component]*int_Cp_over_T_dT)

        rx_stoic_base = abs(rx_stoic[rx_base_component])

        rx_delta_H = (sum(delta_H_formation) + sum(delta_H_Temp_change))/rx_stoic_base
        rx_delta_S = (sum(delta_S_formation) + sum(delta_S_Temp_change))/rx_stoic_base
        rx_delta_G = (rx_delta_H - self.T.val*rx_delta_S)/rx_stoic_base

        # convert to the unit system
        # the themo pkg uses SI, if the user selects another unit system
        # the DVar obj will handle the convertion uppon initialization
        self.delta_H = DVar(rx_delta_H, 'J/mol', self.units) # <- units from the thermo pkg (SI)
        self.delta_S = DVar(rx_delta_S, 'J/mol/K', self.units) # <- units from the thermo pkg (SI)
        self.delta_G = DVar(rx_delta_G, 'J/mol', self.units) # <- units from the thermo pkg (SI)
        try:
            self.equilibrium_constant = exp**(-self.delta_G.val/(self.T.val*self.R.val))
        except Exception as error:
            warnings.warn(f'Something went wrong with the equilibrium constant calculation. Setting it to 1e12 (No reverse reaction). Please check your simulation. Hint: if you are fitting with isothermal = False, try setting it to True. Error: {error}', UserWarning)
            self.equilibrium_constant = 1e12

    def __init__(self, name, stoic, base_component,
                 rate_function, rate_unit, eff_factor = 1, liquid_phase = False,  units = 'SI',
                 T = (25, 'C'), R = (8.314, 'J/mol/K')):
        self.units = units
        self.name = name
        self.stoic = stoic
        self.component_names = list(stoic.keys())
        self.base_component = base_component
        self.rate_function = rate_function
        self.rate_unit = rate_unit
        self.eff_factor = eff_factor
        self.liquid_phase = liquid_phase
        self.T = DVar(*T, self.units)
        self.R = DVar(*R, self.units)

        self.get_thermo_constants()
        self.calc_reaction_HSG()
    def __str__(self):
        string = ""
        for element in self.__dict__:
            if element not in ['thermodynamic_constants', 'Cp_gases_objs', 'Cp_liq_objs']:
                string += str(element) + ": " + str(self.__dict__[element]) + "\n"
        return string