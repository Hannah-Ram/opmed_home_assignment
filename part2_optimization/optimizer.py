"""Greedy optimizer to assign anesthesiologists and operating rooms.

Usage:
  python optimizer.py --input surgeries.csv --output solution.csv

The script reads a surgeries CSV with columns (index/start/end) and
produces a CSV with columns: index,start_time,end_time,anesthetist_id,room_id

Heuristic:
- Sort surgeries by start time
- Greedy assign lowest-numbered available room (20 rooms total)
- Assign each surgery to an existing anesthetist (no overlap) if it
  keeps shift <= 12h and gives minimal marginal cost, otherwise open
  a new anesthetist. Shift duration is from first assigned surgery
  start to last assigned surgery end (gaps allowed). Costs use
  paid hours = max(5, shift_hours) and overtime 0.5×hours beyond 9.
"""
import pandas as pd
import csv
import argparse
from datetime import datetime, timedelta
import heapq
import os
import random
import matplotlib.pyplot as plt



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="surgeries.csv")
    p.add_argument("--output", default="solution.csv")
    return p.parse_args()



def read_surgeries(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader) 

        id_col = headers[0]
        start_col = headers[1]
        end_col = headers[2]

        surgeries = []
        for row in reader:
            if not row or len(row) < 3:
                continue  # skip empty/bad lines

            idx = row[0].strip()
            start_s = row[1].strip()
            end_s = row[2].strip()

           
            st = datetime.fromisoformat(start_s)  
            en = datetime.fromisoformat(end_s)

            surgeries.append({"id": idx, "start": st, "end": en})

    return surgeries


def hours(td):
    return td.total_seconds() / 3600.0


def anesthetist_cost(shift_hours):
    paid = max(5.0, shift_hours)
    overtime = max(0.0, shift_hours - 9.0)
    return paid + 0.5 * overtime

def assign_rooms(surgeries, num_rooms=20):
    # returns list of room_id per surgery (same order as surgeries)
    free_rooms = list(range(num_rooms))
    heapq.heapify(free_rooms)
    occupied = []  # heap of (end_time, room_id)
    room_assignment = []
    for s in surgeries:
        # free rooms whose surgery ended
        while occupied and occupied[0][0] <= s["start"]:
            endt, rid = heapq.heappop(occupied)
            heapq.heappush(free_rooms, rid)
        if not free_rooms:
            raise RuntimeError("More than {} concurrent surgeries; cannot assign rooms".format(num_rooms))
        rid = heapq.heappop(free_rooms)
        heapq.heappush(occupied, (s["end"], rid))
        room_assignment.append(rid)
    return room_assignment


def greedy_assign_anesthetists(surgeries):
    # surgeries must be sorted by start
    anesthetists = []  # list of dicts: {id, first_start, last_end, surgeries[]}

    def try_candidates(s):
        best_delta = float("inf")
        best_idx = None
        for i, a in enumerate(anesthetists):
            if a["last_end"] <= s["start"]:
                new_first = a["first_start"]
                new_last = max(a["last_end"], s["end"])
                new_shift_h = hours(new_last - new_first)
                if new_shift_h > 12.0:
                    continue
                old_shift_h = hours(a["last_end"] - a["first_start"])
                cost_before = anesthetist_cost(old_shift_h)
                cost_after = anesthetist_cost(new_shift_h)
                delta = cost_after - cost_before
                if delta < best_delta:
                    best_delta = delta
                    best_idx = i
        return best_idx, best_delta

    assignments = []  # anesthetist id per surgery
    for s in surgeries:
        idx, delta = try_candidates(s)
        if idx is None:
            # create new anesthetist
            aid = len(anesthetists)
            anesthetists.append({"id": aid, "first_start": s["start"], "last_end": s["end"], "surgeries": [s]})
            assignments.append(aid)
        else:
            a = anesthetists[idx]
            a["last_end"] = max(a["last_end"], s["end"])
            a["surgeries"].append(s)
            assignments.append(a["id"])
    return assignments, anesthetists


def compute_metrics(surgeries, anest_ids, anesthetists):
    """Compute summary metrics for the schedule."""
    total_cost = sum(anesthetist_cost(hours(a["last_end"] - a["first_start"])) for a in anesthetists)
    shifts = [hours(a["last_end"] - a["first_start"]) for a in anesthetists]
    num_anest = len(shifts)
    avg_shift = sum(shifts) / num_anest if num_anest else 0.0
    total_overtime_hours = sum(max(0.0, s - 9.0) for s in shifts)
    
    return {
        "total_cost": round(total_cost, 2),
        "num_anesthetists": num_anest,
        "avg_shift_hours": round(avg_shift, 2),
        "total_overtime_hours": round(total_overtime_hours, 2),
        "num_surgeries": len(surgeries)
    }


def write_solution(path_out, surgeries, anest_ids, room_ids):
    with open(path_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "start_time", "end_time", "anesthetist_id", "room_id"])
        for s, aid, rid in zip(surgeries, anest_ids, room_ids):
            writer.writerow([s["id"], s["start"].isoformat(sep=" "), s["end"].isoformat(sep=" "), f"anesthetist - {aid}", f"room - {rid}"])

def max_simultaneous_surgeries(surgeries):
    # sweep line: +1 at start, -1 at end
    events = []
    for s in surgeries:
        events.append((s["start"], +1))
        events.append((s["end"], -1))

    # sort by time- if same time, process ends (-1) before starts (+1)
    events.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    mx = 0
    for t, delta in events:
        cur += delta
        if cur > mx:
            mx = cur
    return mx

def main():
    args = parse_args()
    inp = args.input
    out = args.output
    if not os.path.isabs(inp):
        inp = os.path.join(os.getcwd(), inp)
    surgeries = read_surgeries(inp)
    surgeries.sort(key=lambda x: x["start"])
    mx = max_simultaneous_surgeries(surgeries)
    print("Max simultaneous surgeries:", mx)

    # assign rooms 
    room_ids = assign_rooms(surgeries, num_rooms=20)

    # assign anesthetists
    anest_ids, anesthetists = greedy_assign_anesthetists(surgeries)
    
    write_solution(out, surgeries, anest_ids, room_ids)

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        "index": [s["id"] for s in surgeries],
        "start_time": [s["start"] for s in surgeries],
        "end_time": [s["end"] for s in surgeries],
        "anesthetist_id": [f"anesthetist - {aid}" for aid in anest_ids],
        "room_id": [f"room - {rid}" for rid in room_ids],
    })
    # Ensure correct dtypes
    df["anesthetist_id"] = df["anesthetist_id"].astype(str)
    df["room_id"] = df["room_id"].astype(str)

    # Plot schedule
    try:
        from plot_day_schedule import plot_day_schedule
        plot_day_schedule(df)
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}", flush=True)
    
    # Compute and display metrics
    metrics = compute_metrics(surgeries, anest_ids, anesthetists)
    print(f"Wrote {out}", flush=True)
    print(f"Total anesthetist cost: {metrics['total_cost']:.2f}", flush=True)
    print(f"Number of anesthetists used: {metrics['num_anesthetists']}", flush=True)
    print(f"Average shift duration (hours): {metrics['avg_shift_hours']:.2f}", flush=True)
    print(f"Total overtime hours: {metrics['total_overtime_hours']:.2f}", flush=True)

    # --- Sensitivity Analysis ---
    print("\n--- Sensitivity Analysis ---")
    # 1. What happens to the total cost if you only have 15 rooms instead of 20?
    try:
        room_ids_15 = assign_rooms(surgeries, num_rooms=15)
        anest_ids_15, anesthetists_15 = greedy_assign_anesthetists(surgeries)
        metrics_15 = compute_metrics(surgeries, anest_ids_15, anesthetists_15)
        print(f"Total cost with 15 rooms: {metrics_15['total_cost']:.2f}")
    except Exception as e:
        print(f"With 15 rooms: {e}")

    # 2. How does cost change if the overtime threshold drops to 8 hours?
    def anesthetist_cost_8hr(shift_hours):
        paid = max(5.0, shift_hours)
        overtime = max(0.0, shift_hours - 8.0)
        return paid + 0.5 * overtime
    total_cost_8hr = sum(anesthetist_cost_8hr(hours(a["last_end"] - a["first_start"])) for a in anesthetists)
    shifts = [hours(a["last_end"] - a["first_start"]) for a in anesthetists]
    total_overtime_8hr = sum(max(0.0, s - 8.0) for s in shifts)
    print(f"Total cost with overtime threshold 8h: {total_cost_8hr:.2f}")
    print(f"Total overtime hours (8h threshold): {total_overtime_8hr:.2f}")


if __name__ == "__main__":
    main()
