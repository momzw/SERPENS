from src.species import Species
import json


def initialize_global_defaults(parameters):
    """Set up default global parameters."""
    parameters.update({'celest': {}})

    try:
        # Read initial defaults from a JSON file
        with open('resources/input_parameters.json', 'r') as f:
            params = json.load(f)

            # Integration-specific parameters
            parameters.update(params[0]['INTEGRATION_SPECIFICS'])
            parameters.update(params[0]['THERMAL_EVAP_PARAMETERS'])

            # Species-specific parameters
            species_dict = {}
            for species in params[1:]:
                for k, v in species.items():
                    # Add to species dict; can access via parameters.get('species')
                    species_dict[k] = Species(**v)
            #parameters.set('species', species_dict)
            #parameters.set('all_species', list(species_dict.values()))
            parameters.set('species', {})
            parameters.set('all_species', [])
    except FileNotFoundError as e:
        print(f"Error: Parameter file not found: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse parameter file: {str(e)}")



class Parameters:
    def __init__(self):
        self.params = {}

    def set(self, key, value):
        """Set a parameter."""
        self.params[key] = value

    def get(self, key, default=None):
        """Get a parameter."""
        return self.params.get(key, default)

    def reset(self):
        """Reset all parameters."""
        self.params.clear()

    def update(self, param_dict):
        """Update multiple parameters at once."""
        for key, value in param_dict.items():
            self.set(key, value)

    def update_celest(self, celestial_name=None):
        """
        Update the 'celest' parameter with data specific to the given celestial name.
        """
        if celestial_name is not None:
            try:
                with open('resources/objects.json', 'r') as f:
                    systems = json.load(f)
                    # Filter the celestial systems based on the name
                    celest_entry = [
                        obj for condition, obj in zip(
                            [s['SYSTEM-NAME'] == f"{celestial_name}" for s in systems], systems
                        ) if condition
                    ]
                    if celest_entry:
                        # Update the 'celest' parameter with the first matching entry
                        self.params['celest'] = celest_entry[0]
                    else:
                        print(f"No celestial system found with name: {celestial_name}")
            except FileNotFoundError as e:
                print(f"Error: File not found: {str(e)}")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse objects file: {str(e)}")

    def as_context(self, temporary_params):
        return ParameterContext(self, temporary_params)


class ParameterContext:
    def __init__(self, parameters, temp_params):
        self.parameters = parameters
        self.temp_params = temp_params
        self.backup = {}

    def __enter__(self):
        for key, value in self.temp_params.items():
            # Backup original value
            if key in self.parameters.params:
                self.backup[key] = self.parameters.params[key]
            self.parameters.set(key, value)

    def __exit__(self, exc_type, exc_value, traceback):
        for key, value in self.backup.items():
            self.parameters.set(key, value)
        for key in self.temp_params.keys():
            if key not in self.backup:
                del self.parameters.params[key]


# Create a globally available shared instance
GLOBAL_PARAMETERS = Parameters()

# Initialize defaults once
initialize_global_defaults(GLOBAL_PARAMETERS)


