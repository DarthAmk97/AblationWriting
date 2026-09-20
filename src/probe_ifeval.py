import ifeval
print([x for x in dir(ifeval) if not x.startswith("_")][:40])
try:
    import inspect
    print(inspect.getfile(ifeval))
except Exception as e:
    print(e)
