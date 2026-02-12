

class SurfaceIdMapper:

    def __id_map__(self, surface):
        surface_id_map = {
            "dry": 0,
            "wet": 1,
            "icy": 2
        }
        return surface_id_map.get(surface.lower(), -1)

    def __init__(self, *args, **kwargs):




    def run(self,surface):
        return self.__id_map__(surface)