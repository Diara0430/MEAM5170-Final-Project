"""Freeze a closure rule on complete screening data, then cross it with the wrist lever."""
import argparse
import csv
from itertools import product
from pathlib import Path

from experiments.box_contact_window import run
from experiments.shape_catch import CONTACT_GRASPS


PROFILES = ("normal","noisy","corrected")
SCREEN = (1121,1122,1123)


def choose(path):
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    methods = ("original", *CONTACT_GRASPS)
    expected = set(product(SCREEN,range(8),PROFILES,methods))
    keys = {(int(r["seed"]),int(r["scenario"]),r["profile"],r["grasp"]) for r in rows}
    if keys!=expected or len(rows)!=len(keys) or any(r["method"]!="pose" for r in rows):
        raise ValueError("Expected all five grasps on identical throws in each observation condition")
    def totals(grasp, field):
        return [sum(int(r[field]) for r in rows if r["grasp"]==grasp and r["profile"]==p) for p in PROFILES]
    base_q = sum(totals("original","qualified"))
    def order(grasp):
        held, q = totals(grasp,"held"), totals(grasp,"qualified")
        return (sum(q)<base_q, -sum(held), -sum(q), -min(held), CONTACT_GRASPS.index(grasp))
    return min(CONTACT_GRASPS, key=order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seed in SCREEN:
        parser.error("Use fresh throws for confirmation")
    name = choose(args.screen)
    options = list(product(("pose","axis_lever"),("original",name+"_shadow",name)))
    run([args.seed], args.profile, options, args.output_dir, frames=True)


if __name__=="__main__":
    main()
