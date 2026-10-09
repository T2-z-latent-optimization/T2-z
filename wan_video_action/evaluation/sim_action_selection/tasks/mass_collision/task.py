from .simulator import CollisionSimulator


def create_simulator(request):
    return CollisionSimulator(request["config"], request["case"], request["source_root"])
