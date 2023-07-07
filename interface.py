"""
This script allows interfacing basic features of SERPENS by guiding the user.
"""

import sys
import json

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


print("Currently implemented exoplanet systems:")
with open('resources/objects.txt') as f:
    objects = f.read().splitlines(True)
    for i, obj in enumerate(objects):
        prop = json.loads(obj)
        print(f"{i + 1}. " + prop["SYSTEM-NAME"])

from_available = query_yes_no("\nDo you want to analyze an already set up exoplanet system?")
if from_available:
    print("Please choose from the above list by entering the associated integer number: ")
    while True:
        system_choice_index = int(input())
        try:
            system_name = json.loads(objects[system_choice_index - 1])['SYSTEM-NAME']
            print(f"Continuing with system {system_name}")
            break
        except IndexError:
            print("Invalid entry. Please try again.")
elif query_yes_no("Do you wish to set up a new system? Else I will exit the program."):
    print("Please enter the name of the system: ")
    new_sys_name = input()
    print("Setting up basic star parameters.")
    print("\t Mass of the star in solar masses: ")
    new_sys_star_mass = float(input()) * MSOL
    print("\t Radius of the star in solar radii: ")
    new_sys_star_radius = float(input()) * RSOL

    print("Setting up basic exoplanet parameters.")
    print("\t Mass of the exoplanet in Jovian masses: ")
    new_sys_planet_mass = float(input()) * MJUP
    print("\t Radius of the exoplanet in Jovian radii: ")
    new_sys_planet_radius = float(input()) * RJUP
    print("\t Semimajor axis of the exoplanet in AU: ")
    new_sys_planet_semimajor = float(input()) * AU

    print("Setting up basic exomoon parameters.")
    print("\t Mass of the exomoon in kg (Press 'ENTER' to adopt Io value): ")
    user_input = input()
    if user_input == "":
        new_sys_moon_mass = 8.8e22
    else:
        new_sys_moon_mass = float(user_input)
    print("\t Radius of the exomoon in km (Press 'ENTER' to adopt Io value): ")
    user_input = input()
    if user_input == "":
        new_sys_moon_radius = 1820e3
    else:
        new_sys_moon_radius = float(user_input)
    print("\t Semimajor axis of the exomoon in planetary radii: ")
    new_sys_moon_semimajor = float(input()) * new_sys_planet_radius

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

