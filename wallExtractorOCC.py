def set_settings(brep):
    if brep:
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_PYTHON_OPENCASCADE, True)
        settings.set(settings.DISABLE_TRIANGULATION, True)
        settings.USE_WORLD_COORDS = True
        settings.set(settings.USE_BREP_DATA, True)
    else:
        settings = ifcopenshell.geom.settings()
        settings.USE_PYTHON_OPENCASCADE = True
        # very weird behaviour here... but this seems to be the only option to activate both triangulation and the generation of normals
        settings.DISABLE_TRIANGULATION = 1
        settings.set(settings.DISABLE_TRIANGULATION, False)
        settings.WELD_VERTICES = False
        settings.NO_NORMALS = False
        settings.GENERATE_UVS = True
        settings.S = False
        # settings.USE_WORLD_COORDS = True
        settings.set(settings.USE_WORLD_COORDS, True)


if include_geometry:
    instance_iterator = ifcopenshell.geom.iterator(self.settings, self.model, multiprocessing.cpu_count(),
                                                   include=all_products)
    instance_iterator.initialize()
    instance_counter = 0
    guid_seen = []
    while True:
        # every 20th instance print the progress
        if instance_counter % 20 == 0:
            print(f"parsing instance {instance_counter}")
        i_geometry = instance_iterator.get()
        if self.brep:
            i_type_ifc: str = i_geometry.data.type
            i_guid_ifc: str = i_geometry.data.guid
        else:
            i_type_ifc: str = i_geometry.type
            i_guid_ifc: str = i_geometry.guid
        # TODO: the elements seem to be duplicated in the ifc file, so we need a special stopping criterion
        if i_guid_ifc in guid_seen:
            if not instance_iterator.next():
                break
            else:
                continue
        else:
            guid_seen.append(i_guid_ifc)
        #instance = BuildingElement(instance_counter, i_guid_ifc, i_type_ifc, geometry=i_geometry, geometry_type="occ_triangulation")
        instance = BuildingElement(instance_counter, i_guid_ifc, i_type_ifc, geometry=i_geometry.geometry, geometry_type="occ_brep")
        instance_counter += 1
        all_instances.append(instance)