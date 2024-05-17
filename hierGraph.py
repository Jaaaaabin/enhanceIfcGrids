import os
import json
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
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
        self.subgraph = None

        self.output_figure_path = figure_path
        self.init_visualization_settings()
        self.read_data_files(json_grid_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls)
        self.process_storey_info()

    def read_data_files(self, json_grid_relationships, json_st_columns, json_st_walls, json_ns_walls, json_ct_walls):
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
    
    def build_hierarchical_data(self, selected_grid_ids=None):
        
        # Default to all grid IDs if none are provided
        if selected_grid_ids is None:
            selected_grid_ids = self.info_grid_relationships.keys()
        
        # * * * * * * * * * * * * * * * * * * * 
        # Directed Edges.
        # Level 0: Grid
        for grid_id, grid_info in self.info_grid_relationships.items():
            if grid_id not in selected_grid_ids:
                continue
            
            grid_node = f"Grid {grid_id}"
            if grid_node not in self.hierarchical_data:
                self.hierarchical_data[grid_node] = {
                    'children': [],
                    'type': grid_info['type'] + '_grid',
                }

            # Level 1: Storey, unique to each grid
            for storey_id, storey_element_ids in self.ids_per_elevation.items():
                matching_storey_ids = set(grid_info['ids']).intersection(set(storey_element_ids))
                if matching_storey_ids:
                    storey_node = f"{grid_node} - Storey {storey_id}"
                    if storey_node not in self.hierarchical_data:
                        self.hierarchical_data[storey_node] = {
                            'children': [],
                            'type': 'storey',
                        }
                    self.hierarchical_data[grid_node]['children'].append(storey_node)
                    
                    # Level 2: Element ID (with type as an attribute)
                    for element_id in matching_storey_ids:
                        self.hierarchical_data[storey_node]['children'].append(element_id)
        
        # * * * * * * * * * * * * * * * * * * * 
        # Directed Edges.
        for grid_id, grid_info in self.info_grid_relationships.items():
            
            # doesn't working yet..
            
            if 'neighbor' in grid_info:
                
                for storey_id, neighbors in grid_info['neighbor'].items():
                    host_storey_grid_node = f"Grid {grid_id} - Storey {storey_id}"
                    self.hierarchical_data[host_storey_grid_node]['neighbor'] = {}
                    
                    for neighbor_grid_id, neighbor_distance in neighbors.items():
                        neighbor_storey_grid_node = f"Grid {neighbor_grid_id} - Storey {storey_id}"
                        self.hierarchical_data[host_storey_grid_node]['neighbor'].update({
                            neighbor_storey_grid_node: neighbor_distance
                        })
    
    def create_hierarchical_graph(self):

        self.graph.clear()
        for node, data in self.hierarchical_data.items():
            self.graph.add_node(node, type=data.get('type', ''))
            
            # hierarchical relations.
            for child in data.get('children', []):
                self.graph.add_node(child, type=self.hierarchical_data[child].get('type', ''))
                self.graph.add_edge(node, child, length=0.)
            
            # neighboring relations.
            if data.get('neighbor', []):
                for neighbor, neighbor_distance in data['neighbor'].items():
                    self.graph.add_edge(node, neighbor, length=neighbor_distance)

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

        label_mapping = {
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
            label=label_mapping.get(etype, etype).replace('_', ' ').title()) for etype in involved_types if etype in self.visualization_settings]

        plt.title("Hierarchical Graph")
        plt.legend(handles=legend_handles, loc='upper right', fontsize='large', handletextpad=0.5, markerscale=1.5)
        plt.savefig(os.path.join(self.output_figure_path, 'hierarchical_graph.png'), dpi=100)
        