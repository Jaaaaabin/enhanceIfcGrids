import json
import os

class JsonFileComparator:

    def __init__(self, working_dir, ifc_a, ifc_b, json_name):
        self.ifc_a = ifc_a
        self.ifc_b = ifc_b
        self.json_name = json_name
        self.file_path_a = os.path.join(working_dir, self.ifc_a, json_name)
        self.file_path_b = os.path.join(working_dir, self.ifc_b, json_name)
        self.working_dir = working_dir

    def read_json_file(self, file_path):
        """
        Reads a JSON file and returns its content.
        """
        with open(file_path, 'r') as file:
            return json.load(file)

    def deep_compare(self, item_a, item_b):
        """Recursively compares two items (dictionaries, lists, or primitives)."""
        if type(item_a) != type(item_b):
            return False
        if isinstance(item_a, dict):
            if item_a.keys() != item_b.keys():
                return False
            return all(self.deep_compare(item_a[key], item_b[key]) for key in item_a)
        elif isinstance(item_a, list):
            if len(item_a) != len(item_b):
                return False
            return all(self.deep_compare(a, b) for a, b in zip(item_a, item_b))
        else:
            return item_a == item_b

    def find_unique_items(self, list_a, list_b):
        """Finds items that are unique to list_a compared to list_b."""
        unique_to_a = []
        for item_a in list_a:
            if not any(self.deep_compare(item_a, item_b) for item_b in list_b):
                unique_to_a.append(item_a)
        return unique_to_a
    
    def compare_json_files(self):
        """Compares two JSON files and identifies differences."""
        data_a = self.read_json_file(self.file_path_a)
        data_b = self.read_json_file(self.file_path_b)

        unique_to_a = self.find_unique_items(data_a, data_b)
        unique_to_b = self.find_unique_items(data_b, data_a)

        return unique_to_a, unique_to_b

    def export_differences(self, unique_to_a, unique_to_b):
        """
        Exports the differences into new JSON files.
        """

        if not bool(unique_to_a):
            output_path_a = os.path.join(self.working_dir, f"EMPTY_{self.ifc_a}_Uni_against_{self.ifc_b}_{self.json_name}")
        else:
            output_path_a = os.path.join(self.working_dir, f"{self.ifc_a}_Uni_against_{self.ifc_b}_{self.json_name}")
        with open(output_path_a, 'w') as file_a:
            json.dump(unique_to_a, file_a, indent=4)

        if not bool(unique_to_b):
            output_path_b = os.path.join(self.working_dir, f"EMPTY_{self.ifc_b}_Uni_against_{self.ifc_a}_{self.json_name}")
        else:
            output_path_b = os.path.join(self.working_dir, f"{self.ifc_b}_Uni_against_{self.ifc_a}_{self.json_name}")
        with open(output_path_b, 'w') as file_b:
            json.dump(unique_to_b, file_b, indent=4)

    def run_comparison(self):
        """
        Runs the comparison process and exports the results.
        """
        unique_to_a, unique_to_b = self.compare_json_files()
        self.export_differences(unique_to_a, unique_to_b)

