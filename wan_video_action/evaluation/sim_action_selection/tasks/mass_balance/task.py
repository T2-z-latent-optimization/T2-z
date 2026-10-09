from .simulator import BalanceSimulator


def create_simulator(request):
    return BalanceSimulator(request["config"], request["case"], request["source_root"])
