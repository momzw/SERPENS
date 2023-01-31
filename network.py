import numpy as np


class Network:

    def __init__(self, id, e_scaling=1):
        self._species_weights = None
        self._network = None
        self._shielded_network = None
        self.e_scaling = e_scaling
        u = 1.660539066e-27
        e = 1.602e-19

        # Na (default network from Io!)
        if id == 1:
            # Na
            self._network = 6 * 60 * 60

        # S (default network from Io!)
        elif id == 2:
            # S
            self._network = 20 * 60 * 60

        # O (default network from Io!)
        elif id == 3:
            self.reaction([["H+", "O+ H", 1/4.5e-6, np.sqrt(2 * 50e3 * e / u)],
                           ["S+", "O+ S", 1 / 3.5e-9],
                           ["S++", "O+ S+",  1 / 1.0e-7],
                           ["e", "O+ 2e", 1 / 5.1e-6],
                           ["y", "O+ e", 1 / 1.7e-8]])

        # O2 (default network from Io!)
        elif id == 4:
            self.reaction([["H+", "O2+ H", 1 / 5.5e-6, np.sqrt(2 * 50e3 * e / u)],
                           ["H+", "O+ O H", 1 / 9.2e-10],
                           ["H+", "O+ O+ H+ 2e", 1 / 7.4e-9],
                           ["H2+", "O2+ H2", 1 / 9.2e-10],
                           ["O+", "O2+ O", 1 / 1.7e-7],
                           ["O+", "O O+ O", 1 / 1.9e-7],
                           ["O+", "O O+ O+ e", 1 / 7.7e-8],
                           ["O+", "O O++ O+ 2e", 1 / 8.0e-9],
                           ["O+", "O+ O+ O+ 2e", 1 / 8.2e-8],
                           ["O+", "O+ O++ O 2e", 1 / 3.9e-8],
                           ["S++", "O2+ S+", 1 / 1.2e-7],
                           ["e", "O O e", 1 / 3.5e-6],
                           ["e", "O2+ 2e", 1 / 5.4e-6],
                           ["e", "O+ O 2e", 1 / 2.0e-6],
                           ["e", "O++ O 3e", 1 / 6.9e-9],
                           ["y", "O O", 1 / 2.0e-7],
                           ["y", "O2+ e", 1 / 3.0e-8],
                           ["y", "O O+ e", 1 / 8.5e-8]])

        # H (default network from Io!)
        elif id == 5:
            # Charge-Exchange with protons at 50keV
            tau4 = 1 / 2.0e-6
            reagent4 = "H+"
            products4 = "H+ H"
            delv4 = np.sqrt(2 * 50e3 * e / u)

            # Charge-Exchange with protons at 60keV
            tau5 = 1 / 9.975e-7
            reagent5 = "H+"
            products5 = "H+ H"
            delv5 = np.sqrt(2 * 60e3 * e / u)

            # Charge-Exchange with protons at 70keV
            tau6 = 1 / 9e-7
            reagent6 = "H+"
            products6 = "H+ H"
            delv6 = np.sqrt(2 * 70e3 * e / u)

            # Charge-Exchange with protons at 80keV
            tau7 = 1 / 6e-7
            reagent7 = "H+"
            products7 = "H+ H"
            delv7 = np.sqrt(2 * 80e3 * e / u)

            tau1 = 1 / 5.4e-7
            reagent1 = "O+"
            products1 = "H+ O"
            delv1 = 0

            tau2 = 1 / 3.0e-6
            reagent2 = "e"
            products2 = "H+ 2e"
            delv2 = 0

            tau3 = 1 / 4.5e-9
            reagent3 = "y"
            products3 = "H+ e"
            delv3 = 0

            lifetimes = np.array([tau1, tau2, tau3, tau4, tau5, tau6, tau7])
            reagents = np.array([reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7])
            products = np.array([products1, products2, products3, products4, products5, products6, products7])
            velocities = np.array([delv1, delv2, delv3, delv4, delv5, delv6, delv7])

            self._network = np.vstack((lifetimes, reagents, products, velocities)).T

        # H2 (default network from Io!)
        elif id == 6:
            self.reaction([["H+", "H H2+", 1 / 4.0e-6],
                           ["O+", "O H2+", 1 / 2.7e-7],
                           ["S++", "S+ H2+", 1 / 1.1e-7],
                           ["e", "2e H2+", 1 / 4.1e-6],
                           ["e", "H+ H 2e", 1 / 2.1e-7],
                           ["e", "H H e", 1 / 1.6e-6],
                           ["y", "H H", 1 / 5.1e-9],
                           ["y", "H2+ e", 1 / 3.1e-9],
                           ["y", "H H+ e", 1 / 6.9e-10]])

        # NaCl (default network from Io!)
        elif id == 7:
            # NaCl
            self._network = 60 * 60 * 24 * 5

        # SO2 (default network from Io!)
        elif id == 8:
            tau1 = 494 * 60 * 60
            reagent1 = "O+"
            products1 = "SO2+ O"

            tau2 = 455473 * 60 * 60
            reagent2 = "S+"
            products2 = "SO2+ S"

            tau3 = 390 * 60 * 60
            reagent3 = "S++"
            products3 = "SO2+ S+"

            tau4 = 61 * 3600
            reagent4 = "e"
            products4 = "SO2+ 2e"

            tau5 = 137 * 3600
            reagent5 = "e"
            products5 = "SO+ O 2e"

            tau6 = 8053 * 3600
            reagent6 = "e"
            products6 = "O+ SO 2e"

            tau7 = 1123 * 3600
            reagent7 = "e"
            products7 = "S+ O2 2e"

            tau8 = 1393 * 3600
            reagent8 = "e"
            products8 = "O2+ S 2e"

            tau9 = 1052300 * 3600
            reagent9 = "e"
            products9 = "O SO++ 3e"

            tau10 = 4093 * 3600
            reagent10 = "y"
            products10 = "SO2+ e"

            tau11 = 45 * 3600
            reagent11 = "y"
            products11 = "SO O"

            tau12 = 131 * 3600
            reagent12 = "y"
            products12 = "S O2"

            lifetimes = np.array(
                [tau1, tau2, tau3, tau4, tau5, tau6, tau7, tau8, tau9, tau10, tau11, tau12])
            reagents = np.array(
                [reagent1, reagent2, reagent3, reagent4, reagent5, reagent6, reagent7, reagent8, reagent9, reagent10,
                 reagent11, reagent12])
            products = np.array(
                [products1, products2, products3, products4, products5, products6, products7, products8, products9,
                 products10, products11, products12])

            self._network = np.vstack((lifetimes, reagents, products)).T

        # O+ (default network from Io!)
        elif id == 9:
            # Charge-Exchange with protons at 50keV
            t1 = 1 / 3.0e-7
            reag1 = "H+"
            prod1 = "H"
            dv1 = np.sqrt(2 * 50e3 * e / (16 * u))

            # Charge-Exchange with protons at 60keV
            t2 = 1 / 2.8e-7
            reag2 = "H+"
            prod2 = "H"
            dv2 = np.sqrt(2 * 60e3 * e / (16 * u))

            # Charge-Exchange with protons at 70keV
            t3 = 1 / 2.0e-7
            reag3 = "H+"
            prod3 = "H"
            dv3 = np.sqrt(2 * 70e3 * e / (16 * u))

            # Charge-Exchange with protons at 80keV
            t4 = 1 / 1.0e-7
            reag4 = "H+"
            prod4 = "H"
            dv4 = np.sqrt(2 * 80e3 * e / (16 * u))

            lifetimes = np.array([t1, t2, t3, t4])
            reagents = np.array([reag1, reag2, reag3, reag4])
            products = np.array([prod1, prod2, prod3, prod4])
            delvs = np.array([dv1, dv2, dv3, dv4])

            self._network = np.vstack((lifetimes, reagents, products, delvs)).T

        # CO2 (default network from WASP-39!)
        elif id == 10:
            # CO2
            self._network = 2.47 * 60

        # K (default network from WASP-39!)
        elif id == 11:
            # K
            self._network = 1.77 * 60

    def reaction(self, reac):
        for r in reac:

            if len(r) == 3:
                r.append(0)

            if r[0] == "e":
                r.insert(0, self.e_scaling * r.pop(2))
            else:
                r.insert(0, r.pop(2))

            if self._network is not None:
                self._network = np.vstack((self._network, np.asarray(r)))
            else:
                self._network = np.asarray(r)

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, tau):
        if isinstance(tau, (float, int)):
            self._network = tau
        else:
            print("Could not set network lifetime")

    @property
    def shielded_network(self):
        return self._shielded_network

    @shielded_network.setter
    def shielded_network(self, tau):
        if isinstance(tau, (float, int)):
            self._shielded_network = tau
        else:
            print("Could not set shielded network lifetime")