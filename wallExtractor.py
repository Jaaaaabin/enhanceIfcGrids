class WallWidthExtractor:
    
    def __init__(self, wall):
        
        if wall is None:
            raise ValueError("Wall must not be None.")
        
        self.wall = wall
        self.representation = None
        self.representation_type = None
        self.width = None

    # note function.
    def get_wall_representation(self):
        #There are two main representations for for wall occurrences if not otherwise specified in the model view definition:

        # IfcWall with IfcMaterialLayerSetUsage is used for all occurrences of walls,
        # that have a non-changing thickness along the wall path and where the thickness parameter can be fully described by a material layer set.
        # These walls are always represented geometrically by an 'Axis' and a 'SweptSolid' shape representation (or by a 'Clipping' geometry based on 'SweptSolid'),
        # if a 3D geometric representation is assigned.
        # The entity IfcWallStandardCase has been deprecated, IfcWall with IfcMaterialLayerSetUsage is used instead.
        # IfcWall without IfcMaterialLayerSetUsage is used for all other occurrences of wall, particularly for walls with changing thickness along the wall path (e.g. polygonal walls),
        # or walls with a non-rectangular cross sections (e.g. L-shaped retaining walls),
        # and walls having an extrusion axis that is unequal to the global Z axis of the project (i.e. non-vertical walls), or walls having only 'Brep', or 'SurfaceModel' geometry,
        # or if a more parametric representation is not intended.

        if not hasattr(self.wall, 'Representation') or not self.wall.Representation or not hasattr(self.wall.Representation, 'Representations') or not self.wall.Representation.Representations:
            raise AttributeError("Wall has no valid Representation or Representations.")
        
        else:

            found_body = False
            for r in self.wall.Representation.Representations:

                if r.RepresentationIdentifier == 'Body':
                    found_body = True
                    self.representation = r
                    if r.Items[0].is_a('IfcExtrudedAreaSolid'):
                        self.representation_type = 'IfcExtrudedAreaSolid'
                    elif r.Items[0].is_a('IfcPolygonalFaceSet'):
                        self.representation_type = 'IfcPolygonalFaceSet'
                    elif r.Items[0].is_a('IfcBooleanClippingResult'):
                        self.representation_type = 'IfcBooleanClippingResult'
                    elif r.Items[0].is_a('IfcFacetedBrep'):
                        self.representation_type = 'IfcFacetedBrep'
                    elif r.Items[0].is_a('IfcAdvancedBrep'):
                        self.representation_type = 'IfcAdvancedBrep'
            if not found_body:
                raise ValueError("No suitable 'Body' representation found.")

    # note function.
    def calculate_width_from_CompositeCurve(self, CompositeCurve):
        
        if CompositeCurve is None:
            raise ValueError("CompositeCurve must not be None.")

        geometry = []
        try:
            if hasattr(CompositeCurve, 'Segments') and len(CompositeCurve.Segments)>1:
                for seg in CompositeCurve.Segments:
                    if hasattr(seg,'ParentCurve') and seg.ParentCurve.is_a('IfcPolyline'):
                        point_coordinates = [p.Coordinates for p in seg.ParentCurve.Points]
                        widths = [abs(a-b) for (a,b) in zip(point_coordinates[:][0],point_coordinates[:][1])]
                        geometry+=widths
                    elif hasattr(seg,'ParentCurve') and seg.ParentCurve.is_a('IfcTrimmedCurve'):
                        pass
            else:
                raise ValueError("Failed to extract geometry for Segments.")
            
            if geometry:
                geometry = [i for i in geometry if abs(i)>0.001]
                return min(geometry)

        except Exception as e:
            raise ArithmeticError(f"Error calculating width from CompositeCurve: {e}")

    # note function.
    def calculate_width_from_sweptArea(self, sweptArea):

        if sweptArea is None:
            raise ValueError("sweptArea must not be None.")

        geometry = []
        try:
            if hasattr(sweptArea, 'XDim') and hasattr(sweptArea, 'YDim'):
                return min(sweptArea.XDim, sweptArea.YDim)
            
            elif hasattr(sweptArea, 'OuterCurve') and sweptArea.OuterCurve.is_a('IfcPolyline'):
                if (hasattr(sweptArea.OuterCurve.Points,'Coordinates') and len(sweptArea.OuterCurve.Points)==1) \
                or len(sweptArea.OuterCurve.Points) > 1:
                    geometry = [p.Coordinates for p in sweptArea.OuterCurve.Points]
                elif hasattr(sweptArea.OuterCurve.Points,'CoordList'):
                    geometry = [p for p in sweptArea.OuterCurve.Points.CoordList]
            elif hasattr(sweptArea, 'OuterCurve') and sweptArea.OuterCurve.is_a('IfcCompositeCurve'):
                return self.calculate_width_from_CompositeCurve(sweptArea.OuterCurve)

            if geometry:
                x_values, y_values = zip(*geometry)
                x_width, y_width = max(x_values) - min(x_values), max(y_values) - min(y_values)
                return min([x_width, y_width])
            else:
                raise ValueError("calculate_width_from_sweptArea: Failed to extract geometry for width calculation.")
        
        except Exception as e:
            raise ArithmeticError(f"calculate_width_from_sweptArea: Error calculating width from sweptArea: {e}")
            
    # note function.
    def calculate_representation_width_from_ExtrudedAreaSolid(self, r):
        
        width_ExtrudedAreaSolid = None
        if r is None:
            raise ValueError("Invalid representation provided.")
    
        if hasattr(r, 'Items') and r.Items:
            try:
                # Ensure r.Items is iterable for both single and multiple items with required attributes
                items = [item for item in (r.Items if len(r.Items) > 1 else [r.Items[0]]) if hasattr(item, 'SweptArea') and hasattr(item, 'Depth')]
                
                if not items:  # If no items have both attributes, raise an error
                    raise ValueError("None of the Items have both 'SweptArea' and 'Depth'.")

                all_widths_Depth = [item.Depth for item in items]
                all_widths_sweptArea = [self.calculate_width_from_sweptArea(item.SweptArea) for item in items]
                
                width_Depth = sum(all_widths_Depth)
                width_sweptArea = sum(all_widths_sweptArea)
                width_ExtrudedAreaSolid = min(width_Depth, width_sweptArea)
                
                return width_ExtrudedAreaSolid
        
            except Exception as e:
                raise ArithmeticError(f"Error in calculate_representation_width_from_ExtrudedAreaSolid: {e}")

    # note function.
    def calculate_width_from_PolygonalFaceSet(self, PolygonalFaceSet):

        if PolygonalFaceSet is None or not hasattr(PolygonalFaceSet, 'Coordinates') or not hasattr(PolygonalFaceSet.Coordinates, 'CoordList'):
            raise ValueError("Invalid PolygonalFaceSet or missing CoordList.")
        
        minY, maxY = float('inf'), float('-inf')
        try:
            for point in PolygonalFaceSet.Coordinates.CoordList:
                y = point[1]
                minY, maxY = min(minY, y), max(maxY, y)
        except IndexError:
            raise ValueError("Error processing coordinates. Ensure CoordList contains points with at least two dimensions.")

        return maxY - minY
    
    # note function.
    def calculate_representation_width_from_PolygonalFaceSet(self, r):

        width_PolygonalFaceSet = None
        if r is None:
            raise ValueError("calculate_representation_width_from_PolygonalFaceSet: Invalid representation provided.")
    
        if hasattr(r, 'Items') and r.Items:
            try:
                # Ensure r.Items is iterable for both single
                items = [item for item in (r.Items if len(r.Items) > 1 else [r.Items[0]]) if item.is_a('IfcPolygonalFaceSet')]
                
                if not items:  # If no items have both attributes, raise an error
                    raise ValueError("calculate_representation_width_from_PolygonalFaceSet: The provided object is not a valid IfcPolygonalFaceSet.")

                all_widths_PolygonalFaceSet = [self.calculate_width_from_PolygonalFaceSet(item) for item in items]
                width_PolygonalFaceSet = min(all_widths_PolygonalFaceSet)
                
                return width_PolygonalFaceSet
        
            except Exception as e:
                raise ArithmeticError(f"calculate_representation_width_from_PolygonalFaceSet: Error in calculate_representation_width_from_PolygonalFaceSet: {e}")
    
    # note function.
    def calculate_width_from_BooleanClippingResult(self, BooleanClippingResult):
        # Validate input
        if BooleanClippingResult is None:
            raise ValueError("BooleanClippingResult is None.")
        if not hasattr(BooleanClippingResult, 'FirstOperand'):
            raise ValueError("BooleanClippingResult is missing FirstOperand attribute.")

        first_operand = BooleanClippingResult.FirstOperand

        # Process IfcExtrudedAreaSolid
        if first_operand.is_a('IfcExtrudedAreaSolid'):
            if hasattr(first_operand, 'SweptArea') and hasattr(first_operand, 'Depth'):
                width_from_ExtrudedAreaSolid = self.calculate_width_from_sweptArea(first_operand.SweptArea)
                width_from_Depth = first_operand.Depth
                return min(width_from_ExtrudedAreaSolid, width_from_Depth)
            else:
                raise ValueError("IfcExtrudedAreaSolid missing required 'SweptArea' or 'Depth'.")

        # Recursively process nested IfcBooleanClippingResult
        elif first_operand.is_a('IfcBooleanClippingResult'):
            return self.calculate_width_from_BooleanClippingResult(first_operand)

        # Handle unexpected operand types
        else:
            raise TypeError(f"Unsupported FirstOperand type: {type(first_operand).__name__}")

    # note function.
    def calculate_representation_width_from_BooleanClippingResult(self, r):
        
        width_BooleanClippingResult = None
        if r is None:
            raise ValueError("calculate_representation_width_from_BooleanClippingResult: Invalid representation provided.")
        
        if hasattr(r, 'Items') and r.Items:
            try:
                # Ensure r.Items is iterable for both single
                items = [item for item in (r.Items if len(r.Items) > 1 else [r.Items[0]]) if item.is_a('IfcBooleanClippingResult')]
                
                if not items:  # If no items have both attributes, raise an error
                    raise ValueError("calculate_representation_width_from_BooleanClippingResult: The provided object is not a valid IfcBooleanClippingResult.")
                
                all_widths_BooleanClippingResult = []
                for item in items:
                    if hasattr(item, 'FirstOperand'):
                        all_widths_BooleanClippingResult.append(self.calculate_width_from_BooleanClippingResult(item))
                    else:
                        raise TypeError("calculate_representation_width_from_BooleanClippingResult: The provided IfcBooleanClippingResult's FirstOperand isn't a IfcExtrudedAreaSolid, nor a IfcBooleanClippingResult.")
                
                width_BooleanClippingResult = min(all_widths_BooleanClippingResult)
                return width_BooleanClippingResult
            
            except Exception as e:
                raise ArithmeticError(f"Error calculating width from IfcBooleanClippingResult: {e}")

    def calculate_representation_width_from_FacetedBrep(self, r):

        if not r:
            raise ValueError("Input 'r' is empty or None.")
        
        width_FacetedBrep = None
        all_widths = []

        if hasattr(r, 'Items') and r.Items:
            try:
                # Ensure r.Items is iterable for both single
                items = [item for item in (r.Items if len(r.Items) > 1 else [r.Items[0]]) if item.is_a('IfcFacetedBrep')]
                
                if not items:  # If no items have both attributes, raise an error
                    raise ValueError("calculate_representation_width_from_FacetedBrep: The provided object is not a valid IfcFacetedBrep.")
                
                for item in items:

                    if not hasattr(item, 'Outer') or not hasattr(item.Outer, 'CfsFaces'):
                        raise AttributeError("Missing expected 'Outer' or 'CfsFaces' attributes in itemresentation.")
                    
                    item_faces = item.Outer.CfsFaces
                    all_widths_item_f = []

                    for item_f in item_faces:
                        if not hasattr(item_f, 'Bounds') or not item_f.Bounds or not hasattr(item_f.Bounds[0], 'Bound') or not hasattr(item_f.Bounds[0].Bound, 'Polygon'):
                            raise AttributeError("calculate_representation_width_from_FacetedBrep: itemresentation face missing 'Bounds', 'Bound', or 'Polygon'.")

                        item_f_points = item_f.Bounds[0].Bound.Polygon
                        coordinates_item_f_points = [p.Coordinates for p in item_f_points]
                        coordinates_item_f_points = list(map(list, zip(*coordinates_item_f_points)))
                        
                        if not coordinates_item_f_points:
                            raise ValueError("calculate_representation_width_from_FacetedBrep: No coordinates found for itemresentation face points.")

                        try:
                            width_item_f = [(max(dimension) - min(dimension)) for dimension in coordinates_item_f_points]
                            # Filter out dimensions with negligible width and find the minimum non-negligible width
                            width_item_f = min([x for x in width_item_f if abs(x) > 0.001])
                        except Exception as calc_err:
                            raise ValueError(f"calculate_representation_width_from_FacetedBrep: Error calculating width for a itemresentation face: {calc_err}")

                        all_widths_item_f.append(width_item_f)

                    if not all_widths_item_f:
                        raise ValueError("calculate_representation_width_from_FacetedBrep: Failed to calculate widths for any itemresentation faces.")

                    width = min(all_widths_item_f)
                    all_widths.append(width)

                if not all_widths:
                    raise ValueError("calculate_representation_width_from_FacetedBrep: Failed to calculate widths for any itemresentations.")

                width_FacetedBrep = max(all_widths)
                return width_FacetedBrep

            except AttributeError as ae:
                raise AttributeError(f"calculate_representation_width_from_FacetedBrep: Attribute error encountered: {ae}")
            except ValueError as ve:
                raise ValueError(f"calculate_representation_width_from_FacetedBrep: Value error encountered: {ve}")
            except Exception as e:
                raise Exception(f"calculate_representation_width_from_FacetedBrep: Unexpected error encountered: {e}")

    # note function.
    def calculate_representation_width_from_AdvancedBrep(self, r):
        
        if not r:
            raise ValueError("Input 'r' is empty or None.")
        
        def tuple_abs_difference(a, b):
            return max(abs(b[0] - a[0]), abs(b[1] - a[1]), abs(b[2] - a[2]))
        
        width_AdvancedBrep = None
        all_widths = []
        
        if hasattr(r, 'Items') and r.Items:
            try:
                # Ensure r.Items is iterable for both single
                items = [item for item in (r.Items if len(r.Items) > 1 else [r.Items[0]]) if item.is_a('IfcAdvancedBrep')]

                for item in items:
                    if not hasattr(item, 'Outer') or not hasattr(item.Outer, 'CfsFaces'):
                        raise AttributeError("Missing expected 'Outer' or 'CfsFaces' attributes in itemresentation.")
                    
                    item_faces = item.Outer.CfsFaces
                    all_widths_item_f = []

                    for item_f in item_faces:

                        if not hasattr(item_f, 'Bounds') or not item_f.Bounds or not hasattr(item_f.Bounds[0], 'Bound') or not hasattr(item_f.Bounds[0].Bound, 'EdgeList'):
                            raise AttributeError("width_AdvancedBitem: itemresentation face missing 'Bounds', 'Bound', or 'EdgeList'.")

                        item_f_edges = item_f.Bounds[0].Bound.EdgeList
                        item_f_edges = [ed.EdgeElement for ed in item_f_edges]
                        width_item_f_edges = max(list([tuple_abs_difference(ed.EdgeEnd.VertexGeometry.Coordinates, ed.EdgeStart.VertexGeometry.Coordinates)
                                            for ed in item_f_edges]))
                        
                        if not width_item_f_edges:
                            raise ValueError("width_AdvancedBrep: No coordinates found for itemresentation face points.")

                        all_widths_item_f.append(width_item_f_edges)

                    if not all_widths_item_f:
                        raise ValueError("width_AdvancedBrep: Failed to calculate widths for any itemresentation faces.")

                    width = min(all_widths_item_f)
                    all_widths.append(width)

                if not all_widths:
                    raise ValueError("width_AdvancedBrep: Failed to calculate widths for any itemresentations.")

                width_AdvancedBrep = min(all_widths)
                return width_AdvancedBrep

            except AttributeError as ae:
                raise AttributeError(f"width_AdvancedBrep: Attribute error encountered: {ae}")
            except ValueError as ve:
                raise ValueError(f"width_AdvancedBrep: Value error encountered: {ve}")
            except Exception as e:
                raise Exception(f"width_AdvancedBrep: Unexpected error encountered: {e}")

    # note function.
    def calc_wall_width_from_representation(self):
        try:

            self.get_wall_representation()
            if not self.representation:
                raise ValueError("Representation extraction failed or no representation found.")

            if self.representation_type == 'IfcExtrudedAreaSolid':
                self.width = self.calculate_representation_width_from_ExtrudedAreaSolid(self.representation) # done.

            elif self.representation_type == 'IfcPolygonalFaceSet':
                self.width = self.calculate_representation_width_from_PolygonalFaceSet(self.representation) # done?

            elif self.representation_type == 'IfcBooleanClippingResult':
                self.width = self.calculate_representation_width_from_BooleanClippingResult(self.representation) # done.

            elif self.representation_type == 'IfcFacetedBrep':
                self.width = self.calculate_representation_width_from_FacetedBrep(self.representation) # done?

            elif self.representation_type == 'IfcAdvancedBrep':
                self.width = self.calculate_representation_width_from_AdvancedBrep(self.representation) # done?
            
            if self.width is None:
                print("stop.")

        except Exception as e:
            raise RuntimeError(f"Failed to calculate wall width: {e}")

    # =================================================
        # width = self.get_wall_width_by_pset(wall)
        # if width == None:
        #     width = self.get_wall_width_by_material(wall)
        #     if width == None:
        #         width = self.get_wall_width_by_geometry(wall)
    #     return width
    # =================================================
    
    # def get_wall_width_by_geometry(self, wall):
        
    #     if not wall.Representation or not wall.Representation.Representations:
            
    #         if wall.is_a('IfcCurtainWall'):
    #             # IfcCurtainWall. Definition from ISO 6707-1:1989:
    #             # Non load bearing wall positioned on the outside of a building and enclosing it.
    #             return 0.0
                
    #             # # -------------------------------------------
    #             # # do we need to validate it as a non-structural wall placed externall.
    #             # for rel_properties in wall.IsDefinedBy:
    #             #     if hasattr(rel_properties.RelatingProertyDefinition, 'HasProerties'):
    #             #         for single_value in rel_properties.RelatingProertyDefinition.HasProerties:
    #             #             if single_value.Name == 'IsExternal' and single_value.NominalValue.wrappedValue == True:
    #             #                 return 0.0
    #             # # -------------------------------------------
    #     else:
    #         print("Wall has no Representation or no Representation.Representations.")
    #         return None
        
    #     # ifc 4 reference view structual view doesn't have any representation.
    #     width = None
    #     for r in wall.Representation.Representations:
    #         if r.RepresentationIdentifier == 'Body':
    #             try:
    #                 if r.Items[0].is_a('IfcExtrudedAreaSolid'):
    #                     width = self.calculate_representation_width_from_ExtrudedAreaSolid(r)
    #                 elif r.Items[0].is_a('IfcPolygonalFaceSet'):
    #                     width = self.calculate_representation_width_from_PolygonalFaceSet(r)
    #                 if width is not None:
    #                     width = round(width, 4)
    #             except AttributeError:
    #                 continue  # Handle missing attributes gracefully
    #     print ("wall_geometry_thickness:",width)
    #     return width
    
#===================================================================================================
#easywallwidthfunctions ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
            
    # def get_wall_width_by_material(self, wall):
        
    #     wall_material = ifcopenshell.util.element.get_material(wall)
    #     # for ifc4. Error processing model test-base-ifc4-rv-s.ifc: entity instance of type 'IFC4.IfcMaterial' has no attribute 'Thickness'

    #     if wall_material:

    #         # IFC Reference Views, the material layer sets are excluded.
    #         if wall_material.is_a('IfcMaterial') and hasattr(wall_material, 'Thickness'):
    #             # If the material is an IfcMaterial entity, append its name and thickness (if available)
    #             wall_material_thickness = round(wall_material.Thickness, 4)
            
    #         elif wall_material.is_a('IfcMaterialLayerSetUsage') and hasattr(wall_material, 'ForLayerSet'):
    #             wall_material_thickness = 0.0
    #             for layer in wall_material.ForLayerSet.MaterialLayers:
    #                 wall_material_thickness += layer.LayerThickness
    #             wall_material_thickness = round(wall_material_thickness, 4)
            
    #         elif wall_material.is_a('IfcMaterialLayerSet') and hasattr(wall_material, 'MaterialLayers'):
    #             wall_material_thickness = 0.0
    #             for layer in wall_material.MaterialLayers:
    #                 wall_material_thickness += layer.LayerThickness
    #             wall_material_thickness = round(wall_material_thickness, 4)
            
    #         else:
    #             # If the material is not an IfcMaterial or a layer set usage, just append its type and None for thickness
    #             wall_material_thickness = None
    #     else:
    #         # If no material is found, append None for both material and thickness
    #         wall_material_thickness = None
        
    #     print ("wall_material_thickness:",wall_material_thickness)
    #     print ("--------------------------------------")
    #     return wall_material_thickness
    

    # def get_wall_width_by_pset(self, wall):
    #     """Gets the width of a wall from its property sets."""
        
    #     # what about the IfcStandardcaseWall?
    #     if not wall.is_a('IfcWall') and not wall.is_a('IfcCurtainWall') and not wall.is_a('IfcStandardcaseWall'):
    #         return None
        
    #     wall_pset = ifcopenshell.util.element.get_psets(wall)

    #     if wall.is_a('IfcWall') or wall.is_a('IfcStandardcaseWall'):
    #         for pset_key in wall_pset:
    #             if 'Width' in wall_pset[pset_key]:
    #                 return round(wall_pset[pset_key]['Width'], 4)
        
    #     elif wall.is_a('IfcCurtainWall'):
    #         try:
    #             wall_component = wall.IsDecomposedBy[0].RelatedObjects[0]
    #             wall_component_pset = ifcopenshell.util.element.get_psets(wall_component)
    #             if 'Dimensions' in wall_component_pset and 'Thickness' in wall_component_pset['Dimensions']:
    #                 return round(wall_component_pset['Dimensions']['Thickness'], 4)
    #         except (IndexError, KeyError):
    #             pass