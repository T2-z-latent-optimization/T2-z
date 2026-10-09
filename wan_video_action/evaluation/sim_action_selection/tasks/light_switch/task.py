from .simulator import LightSwitchSimulator


def create_simulator(request):
    return LightSwitchSimulator(request["config"], request["case"], request["source_root"])
