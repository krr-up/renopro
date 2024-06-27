#script (python)

from pathlib import Path
import json

pred_name2child_idx = json.loads(
    Path("src", "renopro", "data", "pred_name2child_idx.json").read_text("utf-8")
)

def ast_fact2children(fact):
	  try:
        return [fact.arguments[idx] for idx in pred_name2child_idx.get[fact.name]]
    except KeyError:
        return "invalid"

#end.

