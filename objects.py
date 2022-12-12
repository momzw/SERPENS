
__all__ = ['celestial_objects']

def celestial_objects(moon):

    if moon:
        celest = {
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
            "moon": {"m": 4.799e22,     # Europa
                     "a": 6.709e8,
                     "e": 0.009,
                     "inc": 0.0082,
                     "r": 1560800,
                     "primary": 'planet',
                     "hash": 'moon'
                     },
            "add1": {"m": 8.932e22,     # Io
                     "a": 4.217e8,
                     "e": 0.0041,
                     "r": 1821600,
                     "primary": 'planet'
                     },
            "add2": {"m": 1.4819e23,    # Ganymede
                     "a": 1070400000,
                     "e": 0.0013,
                     "inc": 0.00349,
                     "r": 2410300,
                     "primary": 'planet'
                     },
            "add3": {"m": 1.0759e22,    # Callisto
                     "a": 1882700000,
                     "e": 0.0074,
                     "inc": 0.00335,
                     "primary": 'planet'
                     }
        }
    else:
        celest = {
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
                       "hast": 'planet'}
        }
    return celest