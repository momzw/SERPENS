"""
This script allows interfacing basic features of SERPENS by guiding the user.
"""

import sys
import json
from scheduler import SerpensScheduler
from serpens_analyzer import SerpensAnalyzer
from parameters import Parameters, NewParams
from species import Species

print("                .                    .                        .             ")
print("     *   .                  .              .        .   *          .        ")
print("  .         .           .          .        Welcome to          .      .    ")
print("        o                 _____ ______ _____  _____  ______ _   _  _____    ")
print("         .               / ____|  ____|  __ \|  __ \|  ____| \ | |/ ____|   ")
print("          0     .       | (___ | |__  | |__) | |__) | |__  |  \| | (___     ")
print("                 .       \___ \|  __| |  _  /|  ___/|  __| | . ` |\___ \    ")
print(" .          \          . ____) | |____| | \ \| |    | |____| |\  |____) |   ")
print("      .      \   ,      |_____/|______|_|  \_\_|    |______|_| \_|_____/    ")
print("   .          o     .                           .                   .       ")
print("     .         \                 ,                       .                . ")
print("               #\##\#      .                              .        .        ")
print("             #  #O##\###                .                        .          ")
print("   .        #*#  #\##\###                       .                     ,     ")
print("        .   ##*#  #\##\##               .                     .             ")
print("      .      ##*#  #o##\#         .                             ,       .   ")
print("          .     *#  #\#     .                    .             .          , ")
print("                      \          .                         .                ")
print("____^/\___^--____/\____O______________/\/\---/\___________---______________ ")
print("   /\^   ^  ^    ^                  ^^ ^  '\ ^          ^       ---         ")
print("         --           -            --  -      -         ---  __       ^     ")
print("(~ ASCII Art by Robert Casey)  ___--  ^  ^                         --  __    \n \n")

MSOL = 1.988e30
MJUP = 1.898e27
RSOL = 696340e3
RJUP = 69911e3
AU = 149597870700


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def query_simulation():
    print("Currently implemented exoplanet systems:")
    with open('resources/objects.txt') as f:
        objects = f.read().splitlines(True)
        for i, obj in enumerate(objects):
            prop = json.loads(obj)
            print(f"{i + 1}. " + prop["SYSTEM-NAME"])

    # CHOICE OF EXOPLANET SYSTEM
    from_available = query_yes_no("\nDo you want to analyze an already set up exoplanet system?")

    # Take from already set-up systems
    if from_available:
        print("Please choose from the above list by entering the associated integer number: ")
        while True:
            system_choice_index = int(input())
            try:
                system_name = json.loads(objects[system_choice_index - 1])['SYSTEM-NAME']
                print(f"Continuing with system {system_name}.")
                break
            except IndexError:
                print("Invalid entry. Please try again.")

    # Set up new system
    elif query_yes_no("Do you wish to set up a new system? Else I will exit the program."):
        new_sys_name = input("Please enter the name of the system: ")

        print("Setting up basic star parameters.")
        new_sys_star_mass = float(input("\t Mass of the star in solar masses: ")) * MSOL
        new_sys_star_radius = float(input("\t Radius of the star in solar radii: ")) * RSOL

        print("Setting up basic exoplanet parameters.")
        new_sys_planet_mass = float(input("\t Mass of the exoplanet in Jovian masses: ")) * MJUP
        new_sys_planet_radius = float(input("\t Radius of the exoplanet in Jovian radii: ")) * RJUP
        new_sys_planet_semimajor = float(input("\t Semimajor axis of the exoplanet in AU: ")) * AU

        print("Setting up basic exomoon parameters.")
        user_input = input("\t Mass of the exomoon in kg (Press 'ENTER' to adopt Io value): ")
        if user_input == "":
            new_sys_moon_mass = 8.8e22
        else:
            new_sys_moon_mass = float(user_input)
        user_input = input("\t Radius of the exomoon in km (Press 'ENTER' to adopt Io value): ")
        if user_input == "":
            new_sys_moon_radius = 1820e3
        else:
            new_sys_moon_radius = float(user_input)
        new_sys_moon_semimajor = float(input("\t Semimajor axis of the exomoon in planetary radii: ")) * new_sys_planet_radius

        new_sys_dict = {
            'SYSTEM-NAME': new_sys_name,
            'star': {
                'm': new_sys_star_mass,
                'r': new_sys_star_radius,
                'hash': 'star'
            },
            'planet': {
                'm': new_sys_planet_mass,
                'r': new_sys_planet_radius,
                'a': new_sys_planet_semimajor,
                'hash': 'planet',
                'primary': 'star'
            },
            'moon': {
                'm': new_sys_moon_mass,
                'r': new_sys_moon_radius,
                'a': new_sys_moon_semimajor,
                'hash': 'moon',
                'primary': 'planet'
            }
        }
        print("Additonal parameters including eccentricity and inclination may be set inside the corresponding entry in the resources/objects.txt file.")
        with open('resources/objects.txt', 'a') as f:
            f.write("\n")
            f.write(json.dumps(new_sys_dict))
        print(f"New system '{new_sys_name}' saved.")
        system_name = new_sys_name
    else:
        sys.exit()

    # PARAMETER SETTINGS
    params = Parameters()

    print("-------------------")
    print("Parameter Settings:")
    print("Orbit fraction of an advance = " + str(params.int_spec['sim_advance']))
    print("Number of simulation advances = " + str(params.int_spec['num_sim_advances']))
    print(params.species['species1'])

    if query_yes_no("Do you wish to change these parameters "
                    "(if yes, consider doing so in the input_parameters file)?", default="no"):
        new_sim_advance = float(input("Orbit fraction of an advance = "))
        new_num_sim_advances = int(input("Number of simulation advances = "))
        new_species_name = input("Species name (chemical symbol): ")
        new_mass_per_sec = int(input("Mass-loss rate [kg/s] = "))
        new_lifetime = float(input("Lifetime [s] = "))
        new_n_th = int(input("Number of thermal superparticles ejected per advance = "))
        new_n_sp = int(input("Number of sputtered superparticles ejected per advance = "))
        new_beta = float(input("Radiation beta (F_rad/F_grav) = "))

        new_params = NewParams(species=[Species(new_species_name, description=f'Custom {new_species_name}',
                                                n_th=new_n_th, n_sp=new_n_sp, mass_per_sec=new_mass_per_sec,
                                                lifetime=new_lifetime, beta=new_beta)],
                               moon=True,
                               int_spec={"sim_advance": new_sim_advance, "num_sim_advances": new_num_sim_advances},
                               celestial_name=system_name)
        new_params()
        print("NEW parameter Settings:")
        print("Orbit fraction of an advance = " + str(params.int_spec['sim_advance']))
        print("Number of simulation advances = " + str(params.int_spec['num_sim_advances']))
        print(params.species['species1'])
        print("For more advanced settings please see the resources/input_parameters.txt file.")
    print("-------------------")

    save_name = input("Please enter a savename: ")
    save_freq = int(input("Please enter save frequency: "))

    ssch = SerpensScheduler()
    ssch.schedule(save_name, celest_name=system_name)
    ssch.run(save_freq=save_freq)


def query_analysis():
    import matplotlib.pyplot as plt

    serana = SerpensAnalyzer()
    timesteps_to_analyze = [int(x) for x in input("Please enter the timestep(s) you "
                                                  "want to analyze (integers seperated by commas): ").split(",")]
    max_dist_Rp = int(input("Enter the maximal distance away from the planet in units of planetary radii: "))
    print("Note that colormeshing and contouring is currently unavailable")

    while True:
        perspective = input("Do you wish to view the line of sight (enter 'los') "
                            "or a top-down view on the orbital plane (enter 'td')? ")
        if perspective == 'td':
            dimension = int(input("Dimension of density calculation (either '2' or '3'): "))
            add_tri = query_yes_no("Add Delaunay tesselation visualization?", default='no')
            if add_tri:
                add_tri_alpha = float(input("Enter the alpha-value (opacity) of the Delaunay plot: "))
            else:
                add_tri_alpha = 1
            min_dens = float(input("Enter the minimal density in log10 (-10 means 10^-10 as minimal density): "))
            max_dens = float(input("Enter the maximal density in log10 (10 means 10^10 as maximal density): "))

            serana.top_down(timestep=timesteps_to_analyze, d=dimension, colormesh=False, scatter=True, triplot=add_tri,
                            show=True, trialpha=add_tri_alpha, lim=max_dist_Rp,
                            celest_colors=['orange', 'sandybrown', 'red', 'gainsboro', 'tan', 'grey'],
                            colormap=plt.cm.get_cmap("afmhot"), show_moon=True, lvlmin=min_dens, lvlmax=max_dens)
        elif perspective == 'los':
            show_planet = query_yes_no("Draw planet?")
            show_moon = query_yes_no("Draw moon?")

            serana.los(timestep=timesteps_to_analyze, show=True, show_planet=show_planet, show_moon=show_moon,
                       lim=max_dist_Rp, celest_colors=['yellow', 'sandybrown', 'yellow', 'gainsboro', 'tan', 'grey'],
                       scatter=True, colormesh=False, colormap=plt.cm.afmhot)


while True:
    which_query = input("Do you wish to start a simulation (enter '1') "
                        "or analyze the currently present simulation archive file (enter '2')? ")
    if which_query == '1':
        query_simulation()
        break
    elif which_query == '2':
        query_analysis()
        break
    else:
        print("Invalid input. Enter either 1 or 2.")