from .simulator import MultiBackgroundSimulator

isolate_candidate_scenes = True


def create_simulator(request):
    return MultiBackgroundSimulator(request["config"], request["case"], request["source_root"])
