# SolarSquare Solver v33 — Final

A web app that uses Google OR-Tools (CP-SAT constraint solver) to find the **minimum number of vendors** while satisfying these rules:

## Hard constraints (cannot be violated)

1. Each day's DD pairs filled exactly
2. Each day's SD sites filled exactly
3. No vendor does DD on two consecutive days
4. Peak day has zero idle vendors
5. SL recovery only on day after a working day
6. Total monthly slip recoveries = deterministic count
7. **Every individual 2-install vendor profit ≥ ₹0**
8. **Every individual 1-install vendor profit ≥ ₹0**

## Soft target

- Peak day Pxx slip coverage (default P75) — solver tries to hit this but **degrades automatically** if needed to keep all vendors profitable

## Objective

- **Primary:** minimize total vendors
- **Secondary:** maximize the minimum per-vendor profit

## Setup (one-time)

```bash
pip install flask ortools
```

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## Files

- `app.py` — Flask server + CP-SAT model
- `index.html` — UI

Both files must be in the same folder.

## Interpreting output

**Status banner:** vendor count + status code from CP-SAT (`OPTIMAL` is best, `FEASIBLE` means valid but possibly improvable within time limit)

**Pxx target vs achieved:** if achieved is lower, the solver had to degrade coverage to keep vendors profitable. The status shows by how much.

**Min 2i / Min 1i profit:** the LOWEST-earning vendor in each pool. Hard constraints guarantee both ≥ ₹0.

## When solver fails

If "No profitable solution found", your inputs are economically infeasible. Try:
- Lower 2i fixed cost
- Raise S2 slab rate (most sites are S2)
- Raise DD discount (closer to 1.0)
- Lower peak ratio
- Increase solver timeout

## How long does it take?

10-60 seconds typically. Complex scenarios with many vendors may need more — increase "Solver timeout" in the UI.
