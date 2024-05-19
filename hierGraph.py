import os
import copy
import json
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from collections import defaultdict

from quickTools import point_to_line_distance

class HierarchicalGraph:

    def __init__(self, figure_path, json_grid_relationships, json_non_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
        
        self.info_grid_relationships = []
        self.ids_nonbound_elements = []
        self.info_st_columns = []
        self.info_st_walls = []
        self.info_ns_walls = []
        self.info_ct_walls = []

        self.hierarchical_data = {}
        self.graph = nx.DiGraph()
        self.subgraph = None

        self.output_figure_path = figure_path
        
        self.read_data_files(json_grid_relationships, json_non_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls)
        self.init_visualization_settings()

    def read_data_files(self, json_grid_relationships, json_non_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
        
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
        self.ids_nonbound_elements = read_json_file(json_non_relationships)
        self.info_st_columns = read_json_file(json_st_columns)
        self.info_st_walls = read_json_file(json_st_walls)
        self.info_ns_walls = read_json_file(json_ns_walls)
        self.info_ct_walls = read_json_file(json_ct_walls)
    
    def init_visualization_settings(self):
        """
        Initializes visualization settings for various building components.
        """

        self.visualization_settings = {
            'storey':"#cecece",
            'st_column':"#a559aa",
            'st_wall':"#f0c571",
            'ns_wall':"#59a89c",
            'ct_wall':"#59a89c",
            'st_grid':"#e02b35",
            'ns_grid':"#082a54",}

#=========================================================================================
# preparations of graph data ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    def process_storey_info(self):
        # Elements grouped by type
        self.ids_per_element_types = defaultdict(list)
        element_types = {
            'st_column': 'info_st_columns',
            'st_wall': 'info_st_walls',
            'ns_wall': 'info_ns_walls',
            'ct_wall': 'info_ct_walls'
        }
        self.ids_per_element_types = {key: [element['id'] for element in getattr(self, attr)] for key, attr in element_types.items()}

        # Elements grouped by elevation
        self.ids_per_elevation = defaultdict(list)
        all_elements = self.info_st_columns + self.info_st_walls + self.info_ns_walls + self.info_ct_walls
        for element in all_elements:
            self.ids_per_elevation[element['elevation']].append(element['id'])
            element['type'] = next((key for key, value in self.ids_per_element_types.items() if element['id'] in value), None)
            self.hierarchical_data[element['id']] = element

    def find_nephew(self):

        # Initialze the nephew values for grid_relationships.
        for k, v in self.info_grid_relationships.items():
            v['nephew-ids'] = []

        self.info_non_relationships = defaultdict(list)
        for elev, elements_per_elevation in self.ids_per_elevation.items():
            nonbound_elements_per_elevation = set(self.ids_nonbound_elements).intersection(elements_per_elevation)
            self.info_non_relationships[elev] = nonbound_elements_per_elevation

        for non_key, non_value in self.info_non_relationships.items():
            information_grids_per_storey = []
            
            # get all grid information on the storey.
            for grid_key, grid_value in self.info_grid_relationships.items():
                if non_key in grid_value['storey']:
                    grid_location_storey = grid_value['location']
                    [loc.append(non_key) for loc in grid_location_storey]
                    information_grids_per_storey.append([grid_key, grid_location_storey])
            
            # find the closest grid (index_grid_min_distance) to the element location point
            for element_id in non_value:
                
                point_location = self.hierarchical_data[element_id]['location'][0]
                min_point_distance = 1000.
                index_grid_min_distance = None
                for info in information_grids_per_storey:
                    index, grid_line = info[0], info[1]
                    if abs(point_to_line_distance(point_location, grid_line)) <= min_point_distance:
                        index_grid_min_distance = index
                    else:
                        continue

                # write the relation in self.info_grid_relationships.
                if index_grid_min_distance is not None:
                    self.info_grid_relationships[index_grid_min_distance]['nephew-ids'].append(element_id)

    def process_relationships(self):

        self.process_storey_info()
        self.find_nephew()

# preparations of graph data  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#=========================================================================================

#=========================================================================================
# build data for graph  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    
    def calculate_location_relative_distance(self, loc_ref, loc_main):
       
        relative_distance = None
        if len(loc_ref) != 2 or len(loc_main) != 2:
            raise ValueError("Issue on location data format")
        else:
            loc_main_point = loc_main[0]

            if abs(loc_main_point[-1] - loc_ref[0][-1])<0.0001:   
                relative_distance = point_to_line_distance(loc_main_point, loc_ref)
            else:
                raise ValueError("Issue on location data z-coordinates")
        
        return relative_distance

    def build_hierarchical_data(self, selected_grid_ids=None):
        
        # Default to all grid IDs if none are provided
        if selected_grid_ids is None:
            selected_grid_ids = self.info_grid_relationships.keys()
        
        # * * * * * * * * * * * * * * * * * * * 
        # Directed Edges for childrens.
        # Level 0: Grid
        #            |
        #            v
        #       Grid per storey
        for grid_id, grid_info in self.info_grid_relationships.items():
            
            if grid_id not in selected_grid_ids:
                continue
            
            grid_node = f"Grid {grid_id}"
            grid_loc_global =  grid_info['location']
            if grid_node not in self.hierarchical_data:
                self.hierarchical_data[grid_node] = {
                    'children': [],
                    'type': grid_info['type'] + '_grid',
                    'location': grid_loc_global,
                }

            # Level 1: Grid per storey
            #            |
            #            v
            #         Elements
            for storey_id, storey_element_ids in self.ids_per_elevation.items():

                grid_loc_storey = copy.deepcopy(grid_loc_global)
                storey_node = f"{grid_node} - Storey {storey_id}"

                matching_storey_child_ids = set(grid_info['ids']).intersection(set(storey_element_ids))
                if matching_storey_child_ids:
                    if storey_node not in self.hierarchical_data:
                        [loc.append(storey_id) for loc in grid_loc_storey]
                        self.hierarchical_data[storey_node] = {
                            'children': [],
                            'type': 'storey',
                            'location': grid_loc_storey,
                            'nephew': [],
                        }
                    self.hierarchical_data[grid_node]['children'].append(storey_node)
                    
                    # - - - - children
                    # Level 2.1: Children Element ID
                    for element_id in matching_storey_child_ids:

                        self.hierarchical_data[element_id]['status'] = 'children'
                        self.hierarchical_data[storey_node]['children'].append(element_id)
                        relative_distance = self.calculate_location_relative_distance(
                            self.hierarchical_data[storey_node]['location'], self.hierarchical_data[element_id]['location'])
                        self.hierarchical_data[element_id]['host'] = storey_node
                        self.hierarchical_data[element_id]['relative_distance'] = relative_distance
                    
                    # - - - - nephew
                    # Level 2.2: Nephew Element ID
                    # this only happens when there's childer on this stoery.
                    matching_storey_nephew_ids = set(grid_info['nephew-ids']).intersection(set(storey_element_ids))
                    if matching_storey_nephew_ids:
                        for element_id in matching_storey_nephew_ids:
                            
                            self.hierarchical_data[element_id]['status'] = 'nephew'
                            self.hierarchical_data[storey_node]['nephew'].append(element_id)                        
                            relative_distance = self.calculate_location_relative_distance(
                            self.hierarchical_data[storey_node]['location'], self.hierarchical_data[element_id]['location'])
                            self.hierarchical_data[element_id]['host'] = storey_node
                            self.hierarchical_data[element_id]['relative_distance'] = relative_distance
                            
            
        # * * * * * * * * * * * * * * * * * * * 
        # (Bi-)Directed Edges for neighbors.
        # Grid 1 (per storey) - > Grid  2 
        # Grid 1 (per storey) < - Grid  2
        
        for grid_id, grid_info in self.info_grid_relationships.items():
            
            if 'neighbor' in grid_info:
                
                for storey_id, neighbors in grid_info['neighbor'].items():
                    host_storey_grid_node = f"Grid {grid_id} - Storey {storey_id}"
                    self.hierarchical_data[host_storey_grid_node]['neighbor'] = {}
                    
                    for neighbor_grid_id, neighbor_distance in neighbors.items():
                        neighbor_storey_grid_node = f"Grid {neighbor_grid_id} - Storey {storey_id}"
                        self.hierarchical_data[host_storey_grid_node]['neighbor'].update({
                            neighbor_storey_grid_node: neighbor_distance
                        })
    
# build data for graph  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#=========================================================================================

#=========================================================================================
# create graph  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    def create_hierarchical_graph(self):

        self.graph.clear()
        for node, data in self.hierarchical_data.items():
            self.graph.add_node(node, type=data.get('type', ''))
            
            # hierarchical relations - children
            for child in data.get('children', []):
                self.graph.add_node(child, type=self.hierarchical_data[child].get('type', ''))
                self.graph.add_edge(node, child, weight=1.0)
            
            # hierarchical relations - nephew
            for nephew in data.get('nephew', []):
                self.graph.add_node(nephew, type=self.hierarchical_data[nephew].get('type', ''))
                self.graph.add_edge(node, nephew, weight=0.1)

                # todo. add the relative distance to the edges.
                # nx.set_edge_attributes(self.graph, {(node, child): {'relative_distance': 0.2}})
                # print (self.graph.edges[1, 2]["relative_distance"])
            
            # # neighboring relations.
            # if data.get('neighbor', []):
            #     for neighbor, neighbor_distance in data['neighbor'].items():
            #         self.graph.add_edge(node, neighbor, weight=0.0)
            #         nx.set_edge_attributes(self.graph, {(node, neighbor): {'relative_distance': neighbor_distance}})

# create graph  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#=========================================================================================

#=========================================================================================
# visualization of graph  ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓

    def visualize_hierarchical_graph(self, nodes_of_interest=None):

        figx, figy, x_per_grid = 40, 15, 6
        all_connected_nodes = set(nodes_of_interest)
        figx = max(figx, len(all_connected_nodes)*x_per_grid)

        if nodes_of_interest:
            for node in nodes_of_interest:
                all_connected_nodes.update(nx.bfs_tree(self.graph, source=node).nodes)
        self.subgraph = self.graph.subgraph(all_connected_nodes)
        self.visualized_graph = self.subgraph if self.subgraph else self.graph
        
        plt.figure(figsize=(figx, figy))
        pos = graphviz_layout(self.visualized_graph, prog='dot')
        
        # Assign colors to nodes based on element type
        node_colors = [self.visualization_settings.get(self.visualized_graph.nodes[node]['type'], 'gray') for node in self.visualized_graph.nodes]
        edge_labels_by_weights = nx.get_edge_attributes(self.visualized_graph, "weight")

        nx.draw(
            self.visualized_graph,
            pos,
            with_labels=False,
            node_size=1500,
            node_color=node_colors,
            font_size=10,
            font_weight="normal",
            edge_color="gray",
            arrows=True
        )
        nx.draw_networkx_edge_labels(
            self.visualized_graph,
            pos,
            edge_labels=edge_labels_by_weights,
            font_color='navy',
        )

        node_label_mapping = {
            'storey': 'Storey Grid',
            'st_column': 'Structural Column',
            'st_wall': 'Structural Wall',
            'ns_wall': 'Non-Structural Wall',
            'ct_wall': 'Curtain Wall',
            'st_grid': 'Structural Grid',
            'ns_grid': 'Non-Structural Grid'
        }
        
        involved_types = {self.visualized_graph.nodes[node]['type'] for node in self.visualized_graph.nodes}
        legend_handles = [plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=self.visualization_settings[etype], markersize=10,
            label=node_label_mapping.get(etype, etype).replace('_', ' ').title()) for etype in involved_types if etype in self.visualization_settings]

        plt.title("Hierarchical Graph")
        plt.legend(handles=legend_handles, loc='upper right', fontsize='large', handletextpad=0.5, markerscale=1.5)
        plt.savefig(os.path.join(self.output_figure_path, 'hierarchical_graph.png'), dpi=100)
        
# visualization of graph  ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
#=========================================================================================