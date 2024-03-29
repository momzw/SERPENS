
__all__ = ['celestial_objects']


def celestial_objects(moon, set=1):
    m_sol = 1.988e30
    m_jup = 1.898e27
    r_sol = 696340e3
    r_jup = 69911e3
    au = 149597870700

    if isinstance(set, str):
        implemented = ['WASP-49',
                       'HD-189733',
                       'HD-209458',
                       'HAT-P-1',
                       'WASP-39',
                       'WASP-17',
                       'WASP-69',
                       'WASP-96',
                       'XO-2N',
                       'WASP-52']
        try:
            set = implemented.index(set) + 2
        except:
            print(f'System "{set}" has not been set up in the objects file. Implemented systems:')
            print(implemented)

    if moon:
        # Jovian system
        # -------------
        celest1 = {
            "SYSTEM-NAME": "Jupiter",
            "star": {"m": 1.988e30,
                     "hash": 'star',
                     "r": 696340000
                     },
            "planet": {"m": 1.898e27,
                       "a": 7.785e11,
                       "e": 0.0489,
                       "inc": 0.0227,
                       "r": 69911000,
                       "primary": 'star',
                       "hash": 'planet'
                       },
            "moon": {"m": 4.799e22,  # Europa
                     "a": 6.709e8,
                     "e": 0.009,
                     "inc": 0.0082,
                     "r": 1560800,
                     "primary": 'planet',
                     "hash": 'moon'
                     },
            "Io": {"m": 8.932e22,  # Io
                     "a": 4.217e8,
                     "e": 0.0041,
                     "r": 1821600,
                     "primary": 'planet'
                     },
            "Ganymede": {"m": 1.4819e23,  # Ganymede
                     "a": 1070400000,
                     "e": 0.0013,
                     "inc": 0.00349,
                     "r": 2410300,
                     "primary": 'planet'
                     },
            "Callisto": {"m": 1.0759e22,  # Callisto
                     "a": 1882700000,
                     "e": 0.0074,
                     "inc": 0.00335,
                     "primary": 'planet'
                     }
        }

        # WASP-49
        # -------
        celest2 = {
            "SYSTEM-NAME": "WASP-49",
            "star": {"m": 0.72 * m_sol,  # 0.72
                     "hash": 'star',
                     "r": 1.038 * r_sol
                     },
            "planet": {"m": 0.378 * m_jup,
                       "r": 1.115 * r_jup,
                       "a": 0.0378 * au,
                       "hash": 'planet',
                       "primary": 'star'
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.74 * 1.115 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # HD-189733
        # ---------
        celest3 = {
            "SYSTEM-NAME": "HD-189733",
            "star": {"m": 0.8 * m_sol,  # 0.8
                     "r": 0.805 * r_sol,
                     "hash": 'star',
                     },
            "planet": {"m": 1.138 * m_jup,
                       "r": 1.138 * r_jup,
                       "a": 0.031 * au,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 2.11 * 1.138 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # HD-209458
        # ---------
        celest4 = {
            "SYSTEM-NAME": "HD-209458",
            "star": {"m": 1.148 * m_sol,  # 1.148
                     "hash": 'star',
                     "r": 1.203 * r_sol
                     },
            "planet": {"m": 0.69 * m_jup,
                       "r": 1.38 * r_jup,
                       "a": 0.0475 * au,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.93 * 1.38 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # HAT-P-1
        # -------
        celest5 = {
            "SYSTEM-NAME": "HAT-P-1",
            "star": {"m": 1.151 * m_sol,
                     "hash": 'star',
                     "r": 1.174 * r_sol
                     },
            "planet": {"m": 0.525 * m_jup,
                       "a": 0.0556 * au,
                       "r": 1.319 * r_jup,
                       "primary": 'star',
                       "hash": 'planet'
                       },
            "moon": {"m": 8.8e22,
                     "a": 2.28 * 1.319 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # WASP-39
        # -------
        celest6 = {
            "SYSTEM-NAME": "WASP-39",
            "star": {"m": 0.93 * m_sol,  # 0.93
                     "r": .895 * r_sol,
                     "hash": 'star',
                     },
            "planet": {"m": 0.28 * m_jup,
                       "r": 90794840,
                       "a": 0.0486 * au,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.79 * 90794840,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # WASP-17
        # -------
        celest7 = {
            "SYSTEM-NAME": "WASP-17",
            "star": {"m": 1.2 * m_sol,  # 1.2
                     "r": 6.74 * 1.99 * r_jup,
                     "hash": 'star',
                     },
            "planet": {"m": 0.51 * m_jup,
                       "r": 1.99 * r_jup,
                       "a": 8.02 * 6.74 * 1.99 * r_jup,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.28 * 1.99 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # WASP-69
        # -------
        celest8 = {
            "SYSTEM-NAME": "WASP-69",
            "star": {"m": 0.83 * m_sol,  # 0.83
                     "r": 7.48 * 1.06 * r_jup,
                     "hash": 'star',
                     },
            "planet": {"m": 0.26 * m_jup,
                       "r": 1.06 * r_jup,
                       "a": 11.97 * 7.48 * 1.06 * r_jup,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.94 * 1.06 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # WASP-96
        # -------
        celest9 = {
            "SYSTEM-NAME": "WASP-96",
            "star": {"m": 1.06 * m_sol,  # 0.83
                     "r": 8.51 * 1.20 * r_jup,
                     "hash": 'star',
                     },
            "planet": {"m": 0.48 * m_jup,
                       "r": 1.20 * r_jup,
                       "a": 9.28 * 8.51 * 1.20 * r_jup,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 2.10 * 1.20 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # XO-2 N
        # -------
        celest10 = {
            "SYSTEM-NAME": "XO-2 N",
            "star": {"m": 0.98 * m_sol,  # 0.83
                     "r": 9.64 * 0.97 * r_jup,
                     "hash": 'star',
                     },
            "planet": {"m": 0.62 * m_jup,
                       "r": 0.97 * r_jup,
                       "a": 8.23 * 9.64 * 0.97 * r_jup,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 2.07 * 0.97 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

        # WASP-52
        celest11 = {
            "SYSTEM-NAME": "WASP-52",
            "star": {"m": 0.87 * m_sol,  # 0.83
                     "r": 6.06 * 1.27 * r_jup,
                     "hash": 'star',
                     },
            "planet": {"m": 0.62 * m_jup,
                       "r": 1.27 * r_jup,
                       "a": 7.40 * 6.06 * 1.27 * r_jup,
                       "hash": 'planet',
                       "primary": 'star',
                       },
            "moon": {"m": 8.8e22,
                     "a": 1.17 * 1.27 * r_jup,
                     "r": 1820e3,
                     "primary": 'planet',
                     "hash": 'moon'
                     }
        }

    else:
        # 55 Cnc-e
        # --------
        celest1 = {
            "star": {"m": 1.799e30,
                     "hash": 'star',
                     "r": 6.56e8
                     },
            "planet": {"m": 4.77179e25,
                       "a": 2.244e9,
                       "e": 0.05,
                       "inc": 0.00288,
                       "r": 1.196e7,
                       "primary": 'star',
                       "hash": 'planet'}
        }

    try:
        return locals()[f"celest{locals()['set']}"]
    except:
        print("Celestial body set not found. Returning set 1. ")
        return celest1
