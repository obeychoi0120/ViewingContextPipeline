"""Print complete summaries in stable sample order, or selected scene inputs."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=10)
parser.add_argument("--id")
parser.add_argument("--arm")
parser.add_argument("--scene", type=int)
args = parser.parse_args()
packets = [json.loads(line) for line in Path("/tmp/random100_prompt_review_260921_packets.jsonl").read_text().splitlines()]
ids = sorted({p["id"] for p in packets})
for cid in ids[args.start:args.stop] if not args.id else [f"microlens_100k_{args.id}"]:
    selected = [p for p in packets if p["id"] == cid and (not args.arm or p["arm"] == args.arm)]
    if not selected:
        continue
    print("\nCONTENT", cid, "TITLE", (selected[0]["summary"] or {}).get("provenance", {}).get("english_title"))
    for packet in selected:
        print(packet["arm"], "SCENES", len(packet["scenes"]))
        if not args.arm:
            print((packet["summary"] or {}).get("text", "[missing summary]"))
        else:
            for row in packet["scenes"]:
                if args.scene is not None and row["scene_idx"] != args.scene:
                    continue
                print("S" + str(row["scene_idx"]), json.dumps(row.get("graph", row.get("description", row.get("raw_response"))), ensure_ascii=False))
