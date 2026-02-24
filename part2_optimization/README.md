# Surgery Schedule Optimizer - Part 2

## Overview
This optimizer solves the problem of efficiently assigning surgical procedures to operating rooms and anesthesiologists while minimizing total labor costs and resource constraints.

---

## Approach & Why This Optimizer Was Chosen

### Problem Statement
We need to schedule multiple surgeries across a single day, assigning each to:
- An operating room (limited resource)
- An anesthesiologist (must not have overlapping surgeries)

**Objective:** Minimize total anesthesiologist labor costs while respecting constraints.

### The Greedy Algorithm Approach

The **greedy heuristic** was chosen because:

1. **Efficiency**: Optimal assignment is NP-hard; greedy provides good solutions in polynomial time
2. **Scalability**: Works well for realistic hospital schedules (50-100+ surgeries)
3. **Simplicity**: Easy to understand, explain, and modify for hospital operations
4. **Real-world applicability**: Hospitals need fast decisions, not perfect theoretical solutions

### Algorithm Steps

1. **Sort surgeries** by start time (earliest first)
2. **Room Assignment (Greedy)**:
   - For each surgery, assign the lowest-numbered available room
   - Use a heap to efficiently track when rooms become free
   - Raises error if more than 20 simultaneous surgeries occur

3. **Anesthesiologist Assignment (Greedy Cost Minimization)**:
   - For each surgery, consider all existing anesthesiologists
   - Try assigning to the one with minimal **marginal cost increase**
   - Constraint: shift duration must not exceed 12 hours
   - If no existing anesthesiologist works, create a new one
---

## What Was Done

### Input
CSV file with columns: `id`, `start`, `end`
- Each row represents one surgery with start/end timestamps
- Surgeries are in ISO format: `YYYY-MM-DD HH:MM:SS`

### Processing
1. Read and parse all surgeries
2. Sort by start time
3. Assign rooms using greedy room allocation
4. Assign anesthesiologists using cost-minimizing greedy algorithm
5. Compute summary metrics
6. Generate results CSV

### Output
`solution.csv` with columns:
- `index`: Surgery ID
- `start_time`: Surgery start time
- `end_time`: Surgery end time
- `anesthetist_id`: Assigned anesthesiologist
- `room_id`: Assigned operating room

---

## Results Summary

### Key Metrics (Baseline: 20 Rooms, 9-Hour Overtime Threshold)

| Metric | Value |
|--------|-------|
| **Total Paid Hours** | 205.12 |
| **Number of Anesthesiologists** | 24 |
| **Average Shift Duration** | 6.60 hours |
| **Total Overtime Hours** |27.25 hour |

---

## Sensitivity Analysis

### Question 1: What happens to total cost with 15 rooms instead of 20?

**Answer**: The room constraint does NOT affect anesthesiologist costs when surgeries can still be accommodated. Total cost remains 205.12.

**Explanation**:
- Room count doesn’t affect anesthetist assignment in this implementation: anesthetists are assigned using only surgery start/end times; room IDs are not used in the anesthetist scheduling logic.
- The 15-room constraint is not binding for this dataset: at peak, only 15 surgeries run simultaneously, so reducing capacity from 20 rooms to 15 rooms still supports all surgeries without any conflicts or delays.
- **If fewer than 15 rooms were available:** some overlapping surgeries would have to be **pushed to later times** so no more than the available rooms run at once (i.e., reschedule to remove room conflicts).

**Implication**: Hospital can reduce from 20 to 15 operating rooms without cost penalty for this spesific schedule.

---

### Question 2: How does cost change if overtime threshold drops to 8 hours?

**Scenario**: Overtime premium kicks in at 8 hours instead of 9 hours.

**Original (9h threshold)**:
- Total Cost: **205.12**
- Total Overtime: **27.25 hour**

**Modified (8h threshold)**:
- Total Cost: **$210.62**
- Total Overtime: **38.25 hours**

**Explanation**:
- Lowering the overtime threshold from 9h to 8h increases the number of hours billed at the overtime premium. In our results, total overtime increased from 27.25h to 38.25h (+11.00h).
- Because overtime is priced as an extra 0.5× on top of base pay, the total cost increase matches: 0.5 × 11.00 = 5.50, so cost rose from 205.12 to 210.62 (+5.50).
- The schedule/assignments did not change in this sensitivity test- only the overtime rule changed, so the cost difference is purely due to reclassifying more hours as overtime.

---
### Question 3: How close to optimal do you believe your solution is? 

This is a cost-aware greedy heuristic, so it isn’t guaranteed to be globally optimal and may miss improvements that require coordinated swaps. For this relatively small, constrained scheduling problem, it produces a strong feasible baseline quickly and is likely close to optimal in practice, but the exact gap would require comparison against an exact solver.

---

## Technical Details

### Requirements
- Python 3.8+

Python packages:
- pandas
- matplotlib for plotting

### Files
- `optimizer.py`: Main optimization algorithm
- `surgeries.csv`: Input data containing columns: 
- `solution.csv`: Output schedule 
- `plot_day_schedule.py`: Visualization tool

### Running the Optimizer
```bash
python optimizer.py --input surgeries.csv --output solution.csv
```

---

## Future Enhancements
1. **Dynamic Programming**: Test optimal solutions on small datasets
2. **Machine Learning**: Combine predections of surgery duration from part 1 for better scheduling
3. **Staff Preferences**: Incorporate anesthesiologist availability and preferences
4. **Multiple Days**: Extend to multi-day scheduling with continuity constraints
5. **Real-time Optimization**: Re-optimize when surgeries run late
