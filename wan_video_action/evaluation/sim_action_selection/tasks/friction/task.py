from .simulator import FrictionSimulator


def create_simulator(request):
    return FrictionSimulator(request["config"], request["case"], request["source_root"])
