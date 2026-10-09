from .simulator import GravitySimulator


def create_simulator(request):
    return GravitySimulator(request["config"], request["case"], request["source_root"])
