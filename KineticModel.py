from Unit import Unit

class KineticModel:
    def add_reaction(self, reaction):
        # to add a reaction to an object (PBR for now)
        # reaction is an object of the class Reaction
        # *Args e **kwargs permitem que você passe um número não especificado de argumentos para uma função
        # although we convert the rate provided by the user to the
        # internal system, we will throw an Exception when the rate is not
        # in the desired system
        # the program will pass the variables for rate calculation (T, P, x, F)
        # in the selected system ('SI' or 'cgs')
        # so if the user is using 'cgs' in the reactor obj but passes 'mol/s/kg' as rate units
        # he probably is using SI units in his rate function
        # but since 'cgs' is selected, T, P, and F wil be in 'cgs'
        # this forces the user to be conscious of the unit system
        if reaction.rate_unit != Unit("r", self.units):
            raise ValueError(f'Unit {reaction.rate_unit} inconsistent with {self.units}. {self.units} unit for reaction rate is {Unit("r", self.units)}. Hint: the variables provided for rate calculation (T, P, x, F, K) will all be in the {self.units} system, so write your rate function accordingly.')
        self.reactions[reaction.name] =  reaction

    def store_kin_params(self, params):
        # this function puts the minimized parameters back into the params dictionary
        # it must be called after minimization
        self.kin_params_list = params
        i = 0
        for param_name in self.kin_param_dict:
            self.kin_param_dict[param_name] = params[i]
            i += 1

    def __init__(self, name: str, params: dict, bnds: dict = {}, units: str = 'SI'):
        self.name = name
        self.kin_param_dict = params
        self.kin_param_bnds_dict = bnds
        self.kin_params_list = list(params.values())
        # creating bounds list, following order of kin_params_dict
        self.kin_params_bnds_list = []
        for param_name in self.kin_param_dict:
            # if user provides some bnds but not others, leave the not provided ones empty
            if param_name not in self.kin_param_bnds_dict:
                self.kin_params_bnds_list.append((None, None))
            else:
                self.kin_params_bnds_list.append(self.kin_param_bnds_dict[param_name])
        self.units = units
        self.reactions = {}
        self.aic_c = None
        self.aic = None
        self.minimum = None
    def __str__(self):
        string = f"Name:                {self.name}                                        \n"
        string += f"Parameters:         {self.kin_param_dict}                              \n"
        string += f"Parameters' bounds: {self.kin_param_bnds_dict}                         \n"
        string += f"Unit system:        {self.units}                                       \n"
        string += f"Reactions' names:   {[name for name in self.reactions]}                \n"
        string += f"AIC_c:              {self.aic_c}                                       \n"
        string += f"AIC:                {self.aic}                                         \n"
        string += f"Minimum: \n {self.minimum}\n"
        return string