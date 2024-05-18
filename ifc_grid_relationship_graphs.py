import os
from hierGraph import HierarchicalGraph

PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res'

NUM_VISUALIZATION_GRIDS = 2

def get_hierarchical_data(
    work_path,
    ifc_model,
    info_grid_relationships='info_grid_relationships.json',
    info_non_relationships='info_non_relationships.json',
    info_st_columns='info_columns.json',
    info_st_walls='info_st_walls.json',
    info_ns_walls='info_ns_walls.json',
    info_ct_walls='info_ct_walls.json'):

    # initialization
    hierarchical_graph = HierarchicalGraph(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_grid_relationships),
        os.path.join(work_path, ifc_model, info_non_relationships),
        os.path.join(work_path, ifc_model, info_st_columns),
        os.path.join(work_path, ifc_model, info_st_walls),
        os.path.join(work_path, ifc_model, info_ns_walls),
        os.path.join(work_path, ifc_model, info_ct_walls),
        )
    
    return hierarchical_graph

try:

    model_paths = [filename for filename in os.listdir(DATA_FOLDER_PATH) if os.path.isfile(os.path.join(DATA_FOLDER_PATH, filename))]
    
    for model_path in model_paths:

        # for each building model
        hierarchical_graph = get_hierarchical_data(DATA_RES_PATH, model_path)
        hierarchical_graph.process_relationships()
        hierarchical_graph.build_hierarchical_data()
        hierarchical_graph.create_hierarchical_graph()
        selected_grids = [f'Grid {i}' for i in range(1, NUM_VISUALIZATION_GRIDS + 1)]
        hierarchical_graph.visualize_hierarchical_graph(nodes_of_interest=selected_grids)

except Exception as e:
    print(f"Error accessing directory {DATA_RES_PATH}: {e}")

# for pygraphviz installation
# https://pygraphviz.github.io/documentation/stable/install.html#windows-install
# python -m pip install --config-settings="--global-option=build_ext" ^
#                       --config-settings="--global-option=-IC:\Program Files\Graphviz\include" ^
#                       --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" ^
#                       pygraphviz
