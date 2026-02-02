class DVar:
    """ This class stores dimensional variables containing values and units"""
    # this function is to convert a single DVar obj, I will use it to convert Cps and enthalpies
    # inside the solver loop
    # without having to convert all other variables at the same time
    def convert(self, system):
        self.val = self.units[self.unit][f'to_{system}']['f'](self.val)
        self.unit = self.units[self.unit][f'to_{system}']['u']
        self.system = system
        return self

    def __init__(self, value, unit, system):
        # self.units contains all supported units and their conversion functions
        from units_dict import units_dict
        self.units = units_dict

        if unit in self.units.keys():
            self.val = value
            self.unit = unit
            self.system = system
            self.convert(system) # always convert to the current unit system
        else:
            raise TypeError(f'Unit {unit} not supported. Available units are {self.units.keys()}')

    def __str__(self):
        return f"{self.val} {self.unit}"
    def __repr__(self):
        return f"{self.val} {self.unit}"