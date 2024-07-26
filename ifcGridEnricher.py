import ifcopenshell
import os

class IfcSpatialGridEnrichment:

    def __init__(self, model_path, figure_path):

        self.model = ifcopenshell.open(model_path)

        self.ifc_file_name = os.path.basename(model_path)

        self.output_figure_path = figure_path

    def create_grid(self, name, u_axes, v_axes, w_axes=None, levels=None):
        """
        Create an IfcGrid.
        
        :param name: Name of the grid
        :param u_axes: List of U axis labels (e.g., ['A', 'B', 'C'])
        :param v_axes: List of V axis labels (e.g., ['1', '2', '3'])
        :param w_axes: Optional list of W axis labels
        :param levels: List of level elevations this grid appears on. If None, grid exists on all levels.
        :return: Created IfcGrid
        """
        # Create IfcGridAxis entities for each axis
        u_grid_axes = [self._create_grid_axis(label, 'U') for label in u_axes]
        v_grid_axes = [self._create_grid_axis(label, 'V') for label in v_axes]
        w_grid_axes = [self._create_grid_axis(label, 'W') for label in w_axes] if w_axes else None

        # Create the IfcGrid
        grid = self.model.create_entity(
            'IfcGrid',
            GlobalId=ifcopenshell.guid.new(),
            Name=name,
            UAxes=u_grid_axes,
            VAxes=v_grid_axes,
            WAxes=w_grid_axes
        )

        # If levels are specified, create containment relationship
        if levels:
            for level in levels:
                self._relate_grid_to_level(grid, level)

        return grid
    
    def write_to_new_ifc_file(self, new_file_name):
        
        self.model.write(new_file_name)

    def _create_grid_axis(self, label, axis_type):
        """Create an IfcGridAxis entity."""
        return self.model.create_entity(
            'IfcGridAxis',
            AxisTag=label,
            AxisType=axis_type
        )

    def _relate_grid_to_level(self, grid, level):
        """Relate a grid to a specific building level."""
        self.model.create_entity(
            'IfcRelContainedInSpatialStructure',
            GlobalId=ifcopenshell.guid.new(),
            RelatedElements=[grid],
            RelatingStructure=level
        )

    def link_element_to_grid(self, element, grid, link_type='Reference'):
        """
        Link a building element to a grid.
        
        :param element: The building element to link
        :param grid: The grid to link to
        :param link_type: 'Reference' or 'Containment'
        """
        if link_type == 'Reference':
            self._create_reference_link(element, grid)
        elif link_type == 'Containment':
            self._create_containment_link(element, grid)
        else:
            raise ValueError("Invalid link_type. Use 'Reference' or 'Containment'.")

    def _create_reference_link(self, element, grid):
        """Create a reference link between an element and a grid."""
        self.model.create_entity(
            'IfcRelAssociatesClassification',
            GlobalId=ifcopenshell.guid.new(),
            RelatedObjects=[element],
            RelatingClassification=self._create_grid_reference(grid)
        )

    def _create_containment_link(self, element, grid):
        """Create a containment link between an element and a grid."""
        self.model.create_entity(
            'IfcRelContainedInSpatialStructure',
            GlobalId=ifcopenshell.guid.new(),
            RelatedElements=[element],
            RelatingStructure=grid
        )

    def _create_grid_reference(self, grid):
        """Create a classification reference for a grid."""
        return self.model.create_entity(
            'IfcClassificationReference',
            Location=f"Grid:{grid.Name}",
            Identification=grid.GlobalId,
            Name=grid.Name
        )

# # Create a grid on all levels
# all_level_grid = grid_manager.create_grid(
#     name="Main Grid",
#     u_axes=['A', 'B', 'C'],
#     v_axes=['1', '2', '3']
# )

# # Create a grid on specific levels
# levels = [ifc_file.by_type('IfcBuildingStorey')[0]]  # Assuming you have levels defined
# specific_level_grid = grid_manager.create_grid(
#     name="Special Grid",
#     u_axes=['X', 'Y'],
#     v_axes=['4', '5'],
#     levels=levels
# )

# # Link an element to a grid using reference
# element = ifc_file.by_type('IfcWall')[0]  # Assuming you have walls in your model
# grid_manager.link_element_to_grid(element, all_level_grid, link_type='Reference')

# # Link an element to a grid using containment
# column = ifc_file.by_type('IfcColumn')[0]  # Assuming you have columns in your model
# grid_manager.link_element_to_grid(column, specific_level_grid, link_type='Containment')