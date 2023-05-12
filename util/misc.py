
def register(**kvargs):
    for k, v in kvargs.items():
        globals()[k] = v
