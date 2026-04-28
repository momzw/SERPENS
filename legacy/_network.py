import numpy as np


class Network:
    """
    Chemical network of a species.
    Instead of a whole network a single number can be passed as lifetime of a species.
    """
    def __init__(self, id, e_scaling=1):
        """
        Sets the network for a species according to its id.

        Arguments
        ---------
        id : int
            id of the species. Sets the network accordingly.
        e_scaling : float       (default: 1)
            Electron density scaling for reactions involving electrons.
        """
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
            self.reaction([["O+", "H+ O", 1 / 5.4e-7],
                           ["e", "H+ 2e", 1 / 3.0e-6],
                           ["y", "H+ e", 1 / 4.5e-9],
                           ["H+", "H+ H", 1 / 2.0e-6, np.sqrt(2 * 50e3 * e / u)],
                           ["H+", "H+ H", 1 / 9.975e-7, np.sqrt(2 * 60e3 * e / u)],
                           ["H+", "H+ H", 1 / 9e-7, np.sqrt(2 * 70e3 * e / u)],
                           ["H+", "H+ H", 1 / 6e-7, np.sqrt(2 * 80e3 * e / u)]])

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
            self.reaction([["O+", "SO2+ O", 494 * 60 * 60],
                           ["S+", "SO2+ S", 455473 * 60 * 60],
                           ["S++", "SO2+ S+", 390 * 60 * 60],
                           ["e", "SO2+ 2e", 61 * 3600],
                           ["e", "SO+ O 2e", 137 * 3600],
                           ["e", "O+ SO 2e", 8053 * 3600],
                           ["e", "S+ O2 2e", 1123 * 3600],
                           ["e", "O2+ S 2e", 1393 * 3600],
                           ["e", "O SO++ 3e", 1052300 * 3600],
                           ["y", "SO2+ e", 4093 * 3600],
                           ["y", "SO O", 45 * 3600],
                           ["y", "S O2", 131 * 3600]])

        # O+ (default network from Io!)
        elif id == 9:
            self.reaction([["H+", "H", 1 / 3.0e-7, np.sqrt(2 * 50e3 * e / (16 * u))],
                           ["H+", "H", 1 / 2.8e-7, np.sqrt(2 * 60e3 * e / (16 * u))],
                           ["H+", "H", 1 / 2.0e-7, np.sqrt(2 * 70e3 * e / (16 * u))],
                           ["H+", "H", 1 / 1.0e-7, np.sqrt(2 * 80e3 * e / (16 * u))]])

        # CO2 (default network from WASP-39!)
        elif id == 10:
            # CO2
            self._network = 2.47 * 60

        # K (default network from WASP-39!)
        elif id == 11:
            # K
            self._network = 1.77 * 60

    def reaction(self, reac):
        """
        Adds reactions to a species' network.

        Arguments
        ---------
        reac : array-like (shape (N,3) or (N,4))
            Array of reactions each having entries
            ["Reagents", "Products", "Lifetime" (inverse reaction rate) [, "Velocity change"]]
        """
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
            print("Could not set shielded network lifetime.")