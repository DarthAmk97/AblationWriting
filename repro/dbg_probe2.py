import json
d = json.load(open("exports/jmq_case_probe.json", encoding="utf-8"))
print(type(d), len(d))
r = d[0] if isinstance(d, list) else d[list(d.keys())[0]]
print(list(r.keys()) if isinstance(r, dict) else type(r))