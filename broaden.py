
class Gharib_Nezhad_ea_2021:

    def __init__(self, species='AlH'):
        
        if species in ['AlH']:
            self.a_H2 = [+7.6101e-02, -4.3376e-02, +1.9967e-02, +2.4755e-03]
            self.b_H2 = [-5.6857e-01, +2.7436e-01, +3.6216e-02, +1.5350e-05]
            self.a_He = [+4.8630e-02, +2.1731e+03, -2.5351e+02, +3.8607e+01]
            self.b_He = [+4.4644e+04, -4.4438e+03, +6.9659e+02, +4.7331e+00]

        elif species in ['CaH', 'MgH']:
            self.a_H2 = [+8.4022e-02, -8.2171e+03, +4.6171e+02, -7.9708e+00]
            self.b_H2 = [-9.7733e+04, -1.4141e+03, +2.0290e+02, -1.2797e+01]
            self.a_He = [+4.8000e-02, +7.1656e+02, -3.9616e+01, +6.7367e-01]
            self.b_He = [+1.4992e+04, +1.2361e+02, -1.4988e+01, +1.5056e+00]

        elif species in ['CrH', 'FeH', 'TiH']:
            self.a_H2 = [+7.0910e-02, -6.5083e+04, +2.5980e+03, -3.3292e+01]
            self.b_H2 = [-9.0722e+05, -4.3668e+03, +6.1772e+02, -2.4038e+01]
            self.a_He = [+4.2546e-02, -3.0981e+04, +1.2367e+03, -1.5848e+01]
            self.b_He = [-7.1977e+05, -3.4645e+03, +4.9008e+02, -1.9071e+01]

        elif species in ['SiO']:
            self.a_H2 = [+4.7273e-02, -2.7597e+04, +1.1016e+03, -1.4117e+01]
            self.b_H2 = [-5.7703e+05, -2.7774e+03, +3.9289e+02, -1.5289e+01]
            self.a_He = [+2.8364e-02, -6.7705e+03, +2.7027e+02, -3.4634e+00]
            self.b_He = [-2.3594e+05, -1.1357e+03, +1.6065e+02, -6.2516e+00]

        elif species in ['TiO', 'VO']:
            self.a_H2 = [+1.0000e-01, -2.4549e+05, +8.7760e+03, -8.7104e+01]
            self.b_H2 = [-2.3874e+06, +1.6350e+04, +1.7569e+03, -4.1520e+01]
            self.a_He = [+4.0000e-02, -2.8682e+04, +1.0254e+03, -1.0177e+01]
            self.b_He = [-6.9735e+05, +4.7758e+03, +5.1317e+02, -1.2128e+01]

        else:
            raise ValueError(f'Species \"{species}\" not recognised.')
            
    def Pade_equation(self, J, a, b):
        term1 = a[0] + a[1]*J + a[2]*J**2 + a[3]*J**3
        term2 = 1 + b[0]*J + b[1]*J**2 + b[2]*J**3 + b[3]*J**4
        
        return term1 / term2
    
    def gamma_H2(self, J):
        return self.Pade_equation(J, a=self.a_H2, b=self.b_H2)
    
    def gamma_He(self, J):
        return self.Pade_equation(J, a=self.a_He, b=self.b_He)
