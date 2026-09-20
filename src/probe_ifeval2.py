import ifeval, inspect
print(inspect.signature(ifeval.IfEval.__init__) if hasattr(ifeval.IfEval, "__init__") else "no init")
print([m for m in dir(ifeval.IfEval) if not m.startswith("_")][:30])
print([m for m in dir(ifeval.core) if not m.startswith("_")][:30])
