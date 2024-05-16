import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class HierarchicalGraph:

    def __init__(self, figure_path, json_grid_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
        
        self.info_grid_relationships = []
        self.info_st_columns = []
        self.info_st_walls = []
        self.info_ns_walls = []
        self.info_ct_walls = []

        self.hierarchical_data = {}
        self.graph = nx.DiGraph()

        self.out_fig_path = figure_path
        self.read_infos(json_grid_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls)
        self.refine_storey_info()


    def read_infos(self, json_grid_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
        
        def read_json_file(file_path):
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as file:
                        return json.load(file)
                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    return None
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}.")
                    return None
            else:
                return []

        self.info_grid_relationships = read_json_file(json_grid_relationships)
        self.info_st_columns = read_json_file(json_st_columns)
        self.info_st_walls = read_json_file(json_st_walls)
        self.info_ns_walls = read_json_file(json_ns_walls)
        self.info_ct_walls = read_json_file(json_ct_walls)

    def refine_storey_info(self):
        
        # ids_per_element_types.
        self.ids_per_element_types = defaultdict(list)
        element_types = {
            'st_columns': 'info_st_columns',
            'st_walls': 'info_st_walls',
            'ns_walls': 'info_ns_walls',
            'ct_walls': 'info_ct_walls'}
        self.ids_per_element_types = {key: [info['id'] for info in getattr(self, attr)] for key, attr in element_types.items()}

        # ids_per_elevation.
        self.ids_per_elevation = defaultdict(list)
        all_elem_info = self.info_st_columns + self.info_st_walls + self.info_ns_walls + self.info_ct_walls
        for elem in all_elem_info:
            self.ids_per_elevation[elem['elevation']].append(elem['id'])

    def build_hierarchical_data(self):
        
        # level0: grid
        for grid, grid_relationships in self.info_grid_relationships.items():
            
            grid_node = f"Grid {grid}"
            if grid_node not in self.hierarchical_data:
                self.hierarchical_data[grid_node] = {}

            # level1: storey.
            for storey, storey_ids in self.ids_per_elevation.items():
                matching_storey_ids = set(grid_relationships['ids']).intersection(set(storey_ids))
                if matching_storey_ids:
                    storey_node = f"Level {storey}"
                    if storey_node not in self.hierarchical_data[grid_node]:
                        self.hierarchical_data[grid_node][storey_node] = {}
                    
                    # level2: element type.
                    for element_type, type_ids in self.ids_per_element_types.items():
                        matching_type_ids = matching_storey_ids.intersection(set(type_ids))
                        if matching_type_ids:
                            if element_type not in self.hierarchical_data[grid_node][storey_node]:
                                self.hierarchical_data[grid_node][storey_node][element_type] = []
                            self.hierarchical_data[grid_node][storey_node][element_type].extend(matching_type_ids)
    
    def build_hierarchical_graph(self):

        self.build_hierarchical_data()
        
        for grid, storey in self.hierarchical_data.items():
            self.graph.add_node(grid)
            for storey, element_types in storey.items():
                self.graph.add_node(storey)
                self.graph.add_edge(grid, storey)
                for element_type, ids in element_types.items():
                    element_type_node = f"{storey} - {element_type}"
                    self.graph.add_node(element_type_node)
                    self.graph.add_edge(storey, element_type_node)
                    for element_id in ids:
                        self.graph.add_node(element_id)
                        self.graph.add_edge(element_type_node, element_id)

    def visualize_hierarchical_graph(self):
        pos = nx.spring_layout(self.graph)  # Change to other layouts if needed
        plt.figure(figsize=(15, 10))
        nx.draw(self.graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
        plt.title("Hierarchical Graph")
        plt.show()