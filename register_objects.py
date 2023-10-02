"""
This file is meant to be edited.
Running the script will append or overwrite a set of gravitationally acting objects that are defined
in the 'new_celest' dictionary.
"""

R_SOL = 696340e3
M_SOL = 1.988e30
R_JUP = 69911000
M_JUP = 1.898e27
AU = 1.496e11

new_celest = {
    "SYSTEM-NAME": "WASP-171",
    "star": {"m": 1.171 * M_SOL,
             "r": 1.637 * R_SOL
             },
    "planet": {"m": 1.084 * M_JUP,
               "a": 0.0504 * AU,
               "e": 0.0,
               "inc": 0.03,
               "r": 0.988 * R_JUP,
               "primary": 'star',
               },
    "moon": {"m": 1.345e23,
             "a": 1221830e3,
             "e": 0.0292,
             "inc": 0.0029,
             "r": 2574e3,
             "primary": 'planet',
             "source": True
              }
        }


def read_existing_dictionary(filename: str) -> tuple[list, list]:
    """ Reads dictionaries contained in a file and returns a list of them and values associated to the key
    'SYSTEM-NAME'. """
    existing_dicts = []
    existing_systems = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                try:
                    dictionary = eval(line.strip())
                    if isinstance(dictionary, dict):
                        existing_dicts.append(dictionary)
                        existing_systems.append(dictionary['SYSTEM-NAME'])
                except (SyntaxError, ValueError):
                    pass
        return existing_dicts, existing_systems
    except FileNotFoundError:
        return [], []


def append_dictionary_to_file(filename: str, dictionaries: list):
    """ Writes a list of dictionaries to a file."""
    try:
        with open(filename, 'w') as file:
            for dictionary in dictionaries:
                file.write(str(dictionary) + '\n')
        print("Dictionary appended to ", filename)
    except Exception as e:
        print("An error occurred:", e)


def find_and_replace_dictionary(dict_list: list, target_name: str, replacement_dict: dict) -> bool:
    """ Overwrites a dictionary if it has the same SYSTEM-NAME as new_system_dict. """
    for i, dictionary in enumerate(dict_list):
        if dictionary.get("SYSTEM-NAME") == target_name:
            dict_list[i] = replacement_dict
            return True  # Return True if replacement is successful
    return False  # Return False if no matching dictionary is found


def register_system(new_system_dict: dict):
    """ Reads objects.txt file and appends a new system to it or updates an existing system. """

    filename = "resources/objects.txt"

    existing_dicts, existing_systems = read_existing_dictionary(filename)

    if not new_system_dict['SYSTEM-NAME'] in existing_systems:
        existing_dicts.append(new_system_dict)
        append_dictionary_to_file(filename, existing_dicts)
    else:
        print(f"An object-dictionary with this name already exists in {filename}")
        abort = input("Press 'Enter' to continue and overwrite. Press any other key to abort.")
        if not abort:
            replacement_successful = find_and_replace_dictionary(existing_dicts, new_system_dict['SYSTEM-NAME'], new_system_dict)
            if replacement_successful:
                print("Dictionary successfully overwritten.")
            else:
                print("Dictionary not found for the specified name.")
            append_dictionary_to_file(filename, existing_dicts)


if __name__ == "__main__":
    register_system(new_celest)