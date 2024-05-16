import os
import matplotlib.pyplot as plt

# import networkx as nx
# from networkx.drawing.nx_agraph import write_dot, graphviz_layout

from hierGraph import HierarchicalGraph

PROJECT_PATH = r'C:\dev\phd\enrichIFC\enrichIFC'
DATA_FOLDER_PATH = PROJECT_PATH + r'\data\data_test'
DATA_RES_PATH = PROJECT_PATH + r'\res'

def preparation_of_hierarchical_data(
    work_path,
    ifc_model,
    info_grid_relationships='info_grid_relationships.json',
    info_st_columns='info_columns.json',
    info_st_walls='info_st_walls.json',
    info_ns_walls='info_ns_walls.json',
    info_ct_walls='info_ct_walls.json'):

    # initialization
    hierarchical_graph = HierarchicalGraph(
        os.path.join(work_path, ifc_model),
        os.path.join(work_path, ifc_model, info_grid_relationships),
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
        hierarchical_graph = preparation_of_hierarchical_data(DATA_RES_PATH, model_path)
        hierarchical_graph.build_hierarchical_graph()
        hierarchical_graph.visualize_hierarchical_graph()


except Exception as e:
    print(f"Error accessing directory {DATA_RES_PATH}: {e}")

# G = nx.DiGraph()

# G.add_node("ROOT")

# for i in range(5):
#     G.add_node("Child_%i" % i)
#     G.add_node("Grandchild_%i" % i)
#     G.add_node("Greatgrandchild_%i" % i)

#     G.add_edge("ROOT", "Child_%i" % i)
#     G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
#     G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

# # write dot file to use with graphviz
# # run "dot -Tpng test.dot >test.png"
# write_dot(G,'test.dot')

# # same layout using matplotlib with no labels
# plt.title('draw_networkx')
# pos =graphviz_layout(G, prog='dot')
# nx.draw(G, pos, with_labels=False, arrows=True)
# plt.savefig('nx_test.png')



# for pygraphviz installation
# https://pygraphviz.github.io/documentation/stable/install.html#windows-install
# python -m pip install --config-settings="--global-option=build_ext" ^
#                       --config-settings="--global-option=-IC:\Program Files\Graphviz\include" ^
#                       --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" ^
#                       pygraphviz
