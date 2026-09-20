import json
from table_query import MODELS, select_row

rows = [json.loads(l) for l in open("/root/AblationWriting/runs.jsonl") if l.strip()]
print("total", len(rows))
for m, model in MODELS.items():
    rs = [r for r in rows if r["model"] == m]
    best, comment = select_row(m, rows)
    print(model, len(rs), comment, best["run_id"], "R=%+.3f" % best.get("R_L2", 0))
