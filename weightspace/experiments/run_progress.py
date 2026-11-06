"""Shared progress file for the runs in this folder: one bar for the human (progress.py --watch) and one line per result for the chat."""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Progress:
    def __init__(self, plan, first_detail="starting"):
        self.s = dict(
            started_at=time.time(), finished_at=None, stage=plan[0][0], detail=first_detail, done=0, total=sum(p[1] for p in plan),
            work_done=0.0, work_total=float(sum(p[1] * p[2] for p in plan)),
            stages=[dict(name=p[0], done=0, total=p[1], cost=p[2]) for p in plan], results=[], error=None,
        )
        self.write()

    def write(self):
        tmp = os.path.join(HERE, os.environ.get("RUN_PROGRESS_FILE", "progress.json") + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.s, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(HERE, os.environ.get("RUN_PROGRESS_FILE", "progress.json")))

    def tick(self, stage, detail):
        st = self.s["stages"][stage]
        st["done"] += 1
        self.s["done"] += 1
        self.s["work_done"] += st["cost"]
        self.s["stage"], self.s["detail"] = st["name"], detail
        self.write()

    def note(self, stage, detail):
        self.s["stage"], self.s["detail"] = self.s["stages"][stage]["name"], detail
        self.write()

    def result(self, text):
        self.s["results"].append(text)
        print(text, flush=True)
        self.write()

    def finish(self, error=None):
        self.s["finished_at"], self.s["error"], self.s["detail"] = time.time(), error, "finished" if error is None else "failed"
        self.write()
