def Unit(var, sys):
    # given the variable type and unit system return the corresponding unit
    types = {
            'T':      {'SI': 'K',          'cgs': 'K'},
            'P':      {'SI': 'Pa',         'cgs': 'barye'},
            'F':      {'SI': 'mol/s',      'cgs': 'mol/s'},
            'm':      {'SI': 'kg',         'cgs': 'g'},
            'L':      {'SI': 'm',          'cgs': 'cm'},
            'E':      {'SI': 'J',          'cgs': 'erg'},
            'Power':  {'SI': 'W',          'cgs': 'erg/s'},
            'U':      {'SI': 'W/m2/K',     'cgs': 'erg/s/cm2/K'},
            'd':      {'SI': 'kg/m3',      'cgs': 'g/cm3'},
            'Cp':     {'SI': 'J/mol/K',    'cgs': 'erg/mol/K'},
            'e':      {'SI': 'J/mol',      'cgs': 'erg/mol'},
            'r':      {'SI': 'mol/s/kg',   'cgs': 'mol/s/g'},
            'v':      {'SI': 'm3/s',       'cgs': 'cm3/s'},
            'u':      {'SI': 'm/s',        'cgs': 'cm/s'},
        }
    return types[var][sys]