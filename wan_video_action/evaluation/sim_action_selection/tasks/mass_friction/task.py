from .simulator import MassFrictionSimulator


def create_simulator(request):
    return MassFrictionSimulator(request["config"], request["case"], request["source_root"])
