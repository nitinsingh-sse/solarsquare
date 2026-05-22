"""
SolarSquare 2-Install Vendor Solver — v34 (FAST)

Speed fixes from v33:
- NO slab decision variables in CP-SAT (slabs assigned post-hoc)
- CP-SAT only decides activity per vendor per day
- stop_after_first_solution=True (no proving optimality)
- Per-iteration time limit (~5s each)
- Profitability checked post-CP-SAT

Install:  pip install flask ortools
Run:      python app.py
"""

from flask import Flask, request, jsonify, send_from_directory, Response
import os
import math
import traceback as tb_mod
import io
import datetime

try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
    # Register a font that supports ₹ symbol. DejaVu Sans is bundled with most systems.
    DEJAVU_PATHS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        '/Library/Fonts/DejaVuSans.ttf',  # Mac (if installed via Homebrew)
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',  # Mac built-in (has ₹)
        '/Library/Fonts/Arial Unicode.ttf',  # Older Mac
    ]
    DEJAVU_BOLD_PATHS = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/Library/Fonts/DejaVuSans-Bold.ttf',
    ]
    FONT_REGULAR = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'
    for path in DEJAVU_PATHS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('PDFFont', path))
                FONT_REGULAR = 'PDFFont'
                print(f'[pdf] Using font: {path}')
                # Try bold too
                for bpath in DEJAVU_BOLD_PATHS:
                    if os.path.exists(bpath):
                        pdfmetrics.registerFont(TTFont('PDFFont-Bold', bpath))
                        FONT_BOLD = 'PDFFont-Bold'
                        break
                else:
                    FONT_BOLD = 'PDFFont'  # fall back to regular if no bold
                break
            except Exception as e:
                print(f'[pdf] Could not register {path}: {e}')
    else:
        print('[pdf] No Unicode font found. ₹ symbol may not render. Run: brew install --cask font-dejavu')
except ImportError:
    REPORTLAB_OK = False
    FONT_REGULAR = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'
    print('[warning] reportlab not installed. PDF reports will not work.')
    print('[warning] Install with: pip3 install reportlab --break-system-packages')
import random

app = Flask(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_OK = True
    ORTOOLS_ERR = None
except Exception as e:
    ORTOOLS_OK = False
    ORTOOLS_ERR = f'{type(e).__name__}: {e}'


def bell_curve_demand(total, days, peak_ratio):
    if peak_ratio <= 1.001:
        base = total // days
        rem = total - base * days
        arr = [base] * days
        mid = days // 2
        for i in range(rem):
            arr[(i + mid) % days] += 1
        return arr
    mu = (days - 1) / 2
    peak_value = round((total / days) * peak_ratio)
    sigma = days / (2 + (peak_ratio - 1) * 4)
    raw = [math.exp(-0.5 * ((i - mu) / sigma) ** 2) for i in range(days)]
    max_raw = max(raw)
    arr = [round(x * peak_value / max_raw) for x in raw]
    diff = total - sum(arr)
    peak_idx = round(mu)
    order = sorted((i for i in range(days) if i != peak_idx),
                   key=lambda i: (abs(i - mu), i))
    if diff > 0:
        order = order[::-1]
    oi = 0
    safety = 0
    while diff != 0 and safety < 20000:
        t = order[oi % len(order)]
        if diff > 0:
            arr[t] += 1
            diff -= 1
        elif arr[t] > 0:
            arr[t] -= 1
            diff += 1
        oi += 1
        safety += 1
    return arr


def int_distribute(total, props):
    s = sum(props) or 1
    exact = [total * p / s for p in props]
    r = [int(x) for x in exact]
    diff = total - sum(r)
    fracs = sorted([(i, exact[i] - r[i]) for i in range(len(props))],
                   key=lambda t: -t[1])
    for k in range(diff):
        r[fracs[k % len(fracs)][0]] += 1
    return r


def compute_daily_demand(daily, slab_mix, dd_elig_slabs, elig_pct):
    days = len(daily)
    dd_pairs = []
    sd_slabs = []
    for d in range(days):
        slabs = int_distribute(daily[d], slab_mix)
        max_pairs = int(daily[d] * elig_pct // 2)
        pairs = []
        order = [si for si in [1, 0, 2, 3] if dd_elig_slabs[si]]
        for si in order:
            while slabs[si] >= 2 and len(pairs) < max_pairs:
                pairs.append((si, si))
                slabs[si] -= 2
        while len(pairs) < max_pairs:
            avail = [si for si in order if slabs[si] > 0]
            if len(avail) < 2:
                break
            pairs.append((avail[0], avail[1]))
            slabs[avail[0]] -= 1
            slabs[avail[1]] -= 1
        dd_pairs.append(pairs)
        sd_slabs.append(slabs)
    return dd_pairs, sd_slabs


def monte_carlo_slips(pair_count, sd_by_day, sl2, sl1, percentile, runs=500, seed=42):
    rng = random.Random(seed)
    days = len(pair_count)
    dd_sites = [p * 2 for p in pair_count]
    total_slips = round(sum(dd_sites) * sl2 + sum(sd_by_day) * sl1)
    weights = []
    w_sum = 0.0
    for d in range(days - 1):
        w = dd_sites[d] * sl2 + sd_by_day[d] * sl1
        weights.append(w)
        w_sum += w
    if w_sum == 0:
        return {'total_slips': 0, 'expected_sl': [0]*days, 'pxx_sl': [0]*days}
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / w_sum
        cum.append(acc)
    expected_sl = [0.0] * days
    for d in range(days - 1):
        expected_sl[d + 1] = total_slips * weights[d] / w_sum
    daily_samples = [[] for _ in range(days)]
    for _ in range(runs):
        day_counts = [0] * days
        for _s in range(total_slips):
            u = rng.random()
            src = days - 2
            for i, c in enumerate(cum):
                if u <= c:
                    src = i
                    break
            day_counts[src + 1] += 1
        for d in range(days):
            daily_samples[d].append(day_counts[d])
    pxx_sl = []
    for arr in daily_samples:
        arr.sort()
        idx = min(len(arr) - 1, int(len(arr) * percentile))
        pxx_sl.append(arr[idx])
    return {'total_slips': total_slips, 'expected_sl': expected_sl, 'pxx_sl': pxx_sl}


def solve_schedule(v2, v1, days, pair_count, sd_count, peak_day, sl_needed_by_day, total_slips_required, max_working_days, min_working_days, time_limit):
    model = cp_model.CpModel()
    total_v = v2 + v1
    is_dd = [[model.NewBoolVar(f'd_{v}_{d}') for d in range(days)] for v in range(total_v)]
    is_sd = [[model.NewBoolVar(f's_{v}_{d}') for d in range(days)] for v in range(total_v)]
    is_sl = [[model.NewBoolVar(f'l_{v}_{d}') for d in range(days)] for v in range(total_v)]

    for v in range(v2, total_v):
        for d in range(days):
            model.Add(is_dd[v][d] == 0)

    for v in range(total_v):
        for d in range(days):
            model.Add(is_dd[v][d] + is_sd[v][d] + is_sl[v][d] <= 1)

    for d in range(days):
        model.Add(sum(is_dd[v][d] for v in range(v2)) == pair_count[d])
        model.Add(sum(is_sd[v][d] for v in range(total_v)) == sd_count[d])
        # SL: at least sl_needed_by_day (Pxx coverage). May exceed if schedule allows.
        model.Add(sum(is_sl[v][d] for v in range(total_v)) >= sl_needed_by_day[d])

    # HARD CONSTRAINT: total monthly SL must equal the deterministic slip count.
    # All 87 (or whatever) slips must get recovered somewhere in the month.
    model.Add(sum(is_sl[v][d] for v in range(total_v) for d in range(days)) >= total_slips_required)

    for v in range(v2):
        for d in range(days - 1):
            model.Add(is_dd[v][d] + is_dd[v][d + 1] <= 1)

    # Peak day: do NOT force every vendor to work. Spare vendors may idle.
    # (The original strict constraint was infeasible when total_v > peak work)
    # No additional constraint needed beyond the "at most one activity per day" above.

    for v in range(total_v):
        model.Add(is_sl[v][0] == 0)
        for d in range(1, days):
            model.Add(is_sl[v][d] <= is_dd[v][d - 1] + is_sd[v][d - 1])

    # HARD: each vendor max max_working_days working days (DD + SD + SL count as work)
    # AND min min_working_days to force balanced workload distribution.
    for v in range(total_v):
        work_days = sum(is_dd[v][d] + is_sd[v][d] + is_sl[v][d] for d in range(days))
        model.Add(work_days <= max_working_days)
        model.Add(work_days >= min_working_days)

    # BALANCE DD across 2i vendors: each 2i must do at least (avg × 0.85) DD pairs.
    # This prevents one vendor getting 11 DD while another gets 5.
    if v2 > 0:
        total_dd_pairs = sum(pair_count)
        avg_dd_per_v2 = total_dd_pairs / v2
        min_dd_per_v2 = max(0, int(avg_dd_per_v2 * 0.85))
        max_dd_per_v2 = int(-(-total_dd_pairs // v2) + 1)  # ceil + buffer
        for v in range(v2):
            dd_total = sum(is_dd[v][d] for d in range(days))
            model.Add(dd_total >= min_dd_per_v2)
            model.Add(dd_total <= max_dd_per_v2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    solver.parameters.stop_after_first_solution = True

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    DD, SD, SL = 1, 2, 3
    activity = []
    for v in range(total_v):
        row = []
        for d in range(days):
            if solver.Value(is_dd[v][d]) == 1:
                row.append(DD)
            elif solver.Value(is_sd[v][d]) == 1:
                row.append(SD)
            elif solver.Value(is_sl[v][d]) == 1:
                row.append(SL)
            else:
                row.append(0)
        activity.append(row)
    return activity, solver.StatusName(status)


def assign_slabs(activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                 slab_rates_2i, slab_rates_1i, dd_discount):
    total_v = len(activity)
    payouts = [0.0] * total_v
    roster = [[None] * days for _ in range(total_v)]
    DD, SD, SL = 1, 2, 3

    # DD pairs (always done by 2i, use 2i rates with discount)
    for d in range(days):
        dd_vendors = [v for v in range(v2) if activity[v][d] == DD]
        dd_vendors.sort(key=lambda v: payouts[v])
        pair_payouts = []
        for k, (s1, s2) in enumerate(dd_pairs_per_day[d]):
            r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
            pp = max(r1, r2) + min(r1, r2) * dd_discount
            pair_payouts.append((pp, (s1, s2)))
        pair_payouts.sort(key=lambda x: -x[0])
        for i, v in enumerate(dd_vendors):
            if i < len(pair_payouts):
                pp, (s1, s2) = pair_payouts[i]
                roster[v][d] = ('DD', (s1, s2))
                payouts[v] += pp
            else:
                roster[v][d] = ('DD', (1, 1))

    # SD: 2i uses 2i rates, 1i uses 1i rates. Assign slabs to vendors regardless of type
    # but the payout depends on the vendor's type. Greedy: highest slab to lowest-paid vendor.
    for d in range(days):
        sd_vendors = [v for v in range(total_v) if activity[v][d] == SD]
        sd_vendors.sort(key=lambda v: payouts[v])
        sd_flat = []
        for si in range(4):
            sd_flat.extend([si] * sd_slabs_per_day[d][si])
        # Sort slabs by max rate so highest-paying slab goes first
        sd_flat.sort(key=lambda si: -max(slab_rates_2i[si], slab_rates_1i[si]))
        for i, v in enumerate(sd_vendors):
            if i < len(sd_flat):
                slab = sd_flat[i]
                rate = slab_rates_2i[slab] if v < v2 else slab_rates_1i[slab]
                roster[v][d] = ('SD', slab)
                payouts[v] += rate
            else:
                roster[v][d] = ('SD', 1)

    for d in range(days):
        for v in range(total_v):
            if activity[v][d] == SL:
                roster[v][d] = ('SL', None)
            elif activity[v][d] == 0:
                roster[v][d] = ('idle', None)

    return roster, payouts


def optimize_slabs_cpsat(activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                         slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, time_limit, spread_weight=1):
    """Phase B: given fixed activity schedule, find slab assignments that maximize
    minimum vendor profit. Slab decisions are CP-SAT variables here.
    2i vendors use slab_rates_2i, 1i vendors use slab_rates_1i. DD always uses 2i rates."""
    model = cp_model.CpModel()
    total_v = len(activity)
    DD, SD, SL = 1, 2, 3

    # Decision: for each (day, pair-index k), assign to one DD vendor
    # dd_assign[(d, k, v)] = 1 if vendor v gets pair k on day d
    dd_assign = {}
    pair_payouts_per_day = []
    for d in range(days):
        # Vendors doing DD on day d (already decided by activity)
        dd_vendors = [v for v in range(v2) if activity[v][d] == DD]
        pps = []
        for k, (s1, s2) in enumerate(dd_pairs_per_day[d]):
            r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
            pps.append(int(max(r1, r2) + min(r1, r2) * dd_discount))
        pair_payouts_per_day.append(pps)
        # Each pair assigned to exactly one vendor; each vendor gets exactly one pair
        # (Equal counts because activity already decided how many DD on day d)
        if len(dd_vendors) != len(pps):
            # Schedule inconsistency — fallback
            continue
        for k in range(len(pps)):
            for v in dd_vendors:
                dd_assign[(d, k, v)] = model.NewBoolVar(f'ddA_{d}_{k}_{v}')
            model.Add(sum(dd_assign[(d, k, v)] for v in dd_vendors) == 1)
        for v in dd_vendors:
            model.Add(sum(dd_assign[(d, k, v)] for k in range(len(pps))) == 1)

    # SD: each SD vendor on day d gets one slab from the day's pool
    sd_assign = {}
    sd_payouts_per_day = []
    for d in range(days):
        sd_vendors = [v for v in range(total_v) if activity[v][d] == SD]
        flat_slabs = []
        for si in range(4):
            flat_slabs.extend([si] * sd_slabs_per_day[d][si])
        sd_payouts_per_day.append(flat_slabs)
        if len(sd_vendors) != len(flat_slabs):
            continue
        for k in range(len(flat_slabs)):
            for v in sd_vendors:
                sd_assign[(d, k, v)] = model.NewBoolVar(f'sdA_{d}_{k}_{v}')
            model.Add(sum(sd_assign[(d, k, v)] for v in sd_vendors) == 1)
        for v in sd_vendors:
            model.Add(sum(sd_assign[(d, k, v)] for k in range(len(flat_slabs))) == 1)

    # Total payout per vendor
    max_pay = 30 * 60000
    total_payout = [model.NewIntVar(0, max_pay, f'tp_{v}') for v in range(total_v)]
    for v in range(total_v):
        terms = []
        for d in range(days):
            if activity[v][d] == DD and v < v2:
                pps = pair_payouts_per_day[d]
                for k in range(len(pps)):
                    if (d, k, v) in dd_assign:
                        terms.append(dd_assign[(d, k, v)] * pps[k])
            elif activity[v][d] == SD:
                flat = sd_payouts_per_day[d]
                # 2i uses 2i rates, 1i uses 1i rates
                rate_card = slab_rates_2i if v < v2 else slab_rates_1i
                for k in range(len(flat)):
                    if (d, k, v) in sd_assign:
                        terms.append(sd_assign[(d, k, v)] * int(rate_card[flat[k]]))
        if terms:
            model.Add(total_payout[v] == sum(terms))
        else:
            model.Add(total_payout[v] == 0)

    # Objective: equalize profits. Two parts:
    # 1) Maximize the minimum profit (lift the floor)
    # 2) Minimize the spread (max - min) for equal distribution
    # Combined: maximize (2 * min_profit) - (spread_weight * spread)
    # spread_weight=1: balanced (default), spread_weight=3-5: heavy equalization (polish phase)
    min_profit = model.NewIntVar(-max_pay, max_pay, 'min_profit')
    max_profit = model.NewIntVar(-max_pay, max_pay, 'max_profit')
    for v in range(v2):
        model.Add(min_profit <= total_payout[v] - int(cost_2i))
        model.Add(max_profit >= total_payout[v] - int(cost_2i))
    for v in range(v2, total_v):
        model.Add(min_profit <= total_payout[v] - int(cost_1i))
        model.Add(max_profit >= total_payout[v] - int(cost_1i))
    spread = model.NewIntVar(0, 2 * max_pay, 'spread')
    model.Add(spread == max_profit - min_profit)
    model.Maximize(2 * min_profit - spread_weight * spread)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(status)

    # Build roster with optimized slabs
    roster = [[None] * days for _ in range(total_v)]
    payouts = [0.0] * total_v
    for d in range(days):
        for v in range(total_v):
            if activity[v][d] == DD and v < v2:
                pps = pair_payouts_per_day[d]
                for k in range(len(pps)):
                    if (d, k, v) in dd_assign and solver.Value(dd_assign[(d, k, v)]) == 1:
                        s1, s2 = dd_pairs_per_day[d][k]
                        roster[v][d] = ('DD', (s1, s2))
                        payouts[v] += pps[k]
                        break
                if roster[v][d] is None:
                    roster[v][d] = ('DD', (1, 1))
            elif activity[v][d] == SD:
                flat = sd_payouts_per_day[d]
                rate_card = slab_rates_2i if v < v2 else slab_rates_1i
                for k in range(len(flat)):
                    if (d, k, v) in sd_assign and solver.Value(sd_assign[(d, k, v)]) == 1:
                        slab = flat[k]
                        roster[v][d] = ('SD', slab)
                        payouts[v] += rate_card[slab]
                        break
                if roster[v][d] is None:
                    roster[v][d] = ('SD', 1)
            elif activity[v][d] == SL:
                roster[v][d] = ('SL', None)
            else:
                roster[v][d] = ('idle', None)
    return (roster, payouts), solver.StatusName(status)


def find_optimal_fast(daily, dd_pairs_per_day, sd_slabs_per_day, pair_count, sd_count,
                      peak_day, sl2, sl1, target_pxx, slab_rates_2i, slab_rates_1i, dd_discount,
                      cost_2i, cost_1i, time_limit_sec, max_working_days):
    days = len(daily)
    # lb_v2: max of consecutive DD pairs (no consec constraint) and peak DD
    lb_v2 = max((pair_count[d] + pair_count[d + 1] for d in range(days - 1)), default=0)
    lb_v2 = max(lb_v2, pair_count[peak_day])

    # lb_v1_base: at any day d, total_v must >= pair_count[d] + sd_count[d]
    # (peak day will be tightest, but check all days to be safe)
    # min v1 = max over all days of (pair_count[d] + sd_count[d] - lb_v2)
    # but only counts 2i above pair_count[d] as available for SD
    # Actually simpler: total >= max(pair[d] + sd[d]) for all d
    # So min total = max over d. Then min v1 = max(0, min_total - lb_v2)
    min_total_for_capacity = max(pair_count[d] + sd_count[d] for d in range(days))

    # Also: total vendor-days needed = DD + SD + total_slips. Available = total_v × max_working_days.
    target_total_slips = round(sum(p*2 for p in pair_count) * sl2 + sum(sd_count) * sl1)
    total_workday_demand = sum(pair_count) + sum(sd_count) + target_total_slips
    min_total_from_workdays = -(-total_workday_demand // max_working_days)  # ceil division
    min_total = max(min_total_for_capacity, min_total_from_workdays)
    lb_v1_base = max(0, min_total - lb_v2)
    print(f'[solver] Lower bounds: v2>={lb_v2}, v1>={lb_v1_base}, min_total={min_total} (capacity={min_total_for_capacity}, workdays={min_total_from_workdays}, total_slips={target_total_slips})')

    pxx_levels = [target_pxx]
    for p in [0.50, 0.25, 0.10, 0.0]:
        if p < target_pxx and p not in pxx_levels:
            pxx_levels.append(p)

    # Wall-clock cap: stop searching after time_limit_sec
    import time
    search_start = time.time()
    HARD_CAP = float(time_limit_sec)
    print(f'[solver] Hard cap: {HARD_CAP}s wall clock')

    # Tight budgets to fit within HARD_CAP. Phase A: 3s feasibility, Phase B: 12s optimization.
    per_call_time = 3

    # Track best partial solution in case no fully-profitable exists
    best_partial = None
    best_partial_score = float('inf')

    # Collect ALL profitable candidates, then pick best at end
    profitable_candidates = []
    iterations_since_improvement = 0  # for smart early stop
    done = False  # flag to break out of nested loops

    # Track every iteration for the diagnostic UI table
    all_attempts = []  # list of {v2, v1, total, status, min_profit, all_profitable}

    iter_count = 0
    # Wider search ranges so we explore all reasonable vendor counts
    for total_extra in range(0, 25):
        if done:
            break
        # Prefer adding 1i (cheaper) before 2i
        for v1_extra in range(0, total_extra + 1):
            if done:
                break
            v2_extra = total_extra - v1_extra
            v2 = lb_v2 + v2_extra
            v1 = lb_v1_base + v1_extra

            # Pre-flight: check basic capacity
            # Every day needs at least pair_count[d] + sd_count[d] vendors working
            min_capacity_per_day = [pair_count[d] + sd_count[d] for d in range(days)]
            max_needed = max(min_capacity_per_day)
            if v2 + v1 < max_needed:
                # Not enough vendors to cover daily demand. Skip all Pxx levels.
                continue
            # 2i vendors must cover DD on every day (only they can do DD)
            if v2 < max(pair_count):
                continue
            # No-consec-DD: need enough 2i for consecutive days
            max_consec_dd = max((pair_count[d] + pair_count[d + 1] for d in range(days - 1)), default=0)
            if v2 < max_consec_dd:
                continue

            # Pre-flight: enough SL capacity? Sum of slack across days 1..29 must >= total_slips.
            # (Day 0 can't host SL because no prior workday.)
            total_slack = sum(max(0, (v2 + v1) - pair_count[d] - sd_count[d]) for d in range(1, days))
            target_total_slips = round(sum(p*2 for p in pair_count) * sl2 + sum(sd_count) * sl1)
            if total_slack < target_total_slips:
                print(f'[solver] skip v2={v2}, v1={v1}: total slack {total_slack} < required slips {target_total_slips}')
                continue

            # Try Pxx levels until one works (lower Pxx = less SL needed = easier)
            iteration_found_any = False
            for pxx in pxx_levels:
                iter_count += 1
                # Wall-clock cap
                elapsed = time.time() - search_start
                if elapsed > HARD_CAP:
                    print(f'[solver] Wall-clock cap reached: {elapsed:.1f}s > {HARD_CAP}s')
                    done = True
                    break
                # Smart early stop: if we have profitable candidates and N more iterations
                # haven't improved the best one, stop. Adding more vendors only hurts profit.
                if profitable_candidates and iterations_since_improvement >= 8:
                    print(f'[solver] Early stop: {iterations_since_improvement} iterations without improvement')
                    done = True
                    break
                mc = monte_carlo_slips(pair_count, sd_count, sl2, sl1, pxx)
                sl_needed = []
                feas = True
                for d in range(days):
                    slack = v2 + v1 - pair_count[d] - sd_count[d]
                    pxx_val = mc['pxx_sl'][d]
                    if slack < pxx_val:
                        feas = False
                        break
                    expected = mc['expected_sl'][d]
                    if d == 0:
                        sl_needed.append(0)
                    elif slack < pxx_val + 1:
                        sl_needed.append(pxx_val)
                    else:
                        sl_needed.append(min(slack, round(expected)))
                if not feas:
                    # Pxx too high for this vendor count — but lower Pxx might work
                    continue
                while sum(sl_needed) < mc['total_slips']:
                    bumped = False
                    for d in sorted(range(days), key=lambda i: -mc['pxx_sl'][i]):
                        slack = v2 + v1 - pair_count[d] - sd_count[d]
                        if sl_needed[d] < slack:
                            sl_needed[d] += 1
                            bumped = True
                            break
                    if not bumped:
                        break
                # min_working_days: force balanced workload.
                # Total work = DD + SD + SL = sum(pair_count) + sum(sd_count) + sum(sl_needed)
                # Min per vendor = floor((total_work - max_slack) / total_v)
                # where max_slack lets some vendors work slightly more than others.
                # Simpler: aim for ~90% of avg, capped at max_working_days
                total_work = sum(pair_count) + sum(sd_count) + sum(sl_needed)
                avg_work = total_work / (v2 + v1)
                min_working_days = max(0, int(avg_work * 0.9))
                # Don't exceed max
                min_working_days = min(min_working_days, max_working_days - 2)
                print(f'[solver] iter {iter_count}: v2={v2}, v1={v1}, pxx={pxx}, target_total_sl={mc["total_slips"]}, min_work={min_working_days}, max_work={max_working_days}')
                activity, status = solve_schedule(v2, v1, days, pair_count, sd_count,
                                                  peak_day, sl_needed, mc['total_slips'], max_working_days, min_working_days, per_call_time)
                if activity is None:
                    print(f'[solver]   CP-SAT: {status}')
                    # Record this attempt
                    all_attempts.append({
                        'v2': v2, 'v1': v1, 'total': v2 + v1, 'pxx': pxx,
                        'status': status.lower(), 'min_profit': None, 'all_profitable': False,
                    })
                    # INFEASIBLE here likely means SL slots too tight. Try lower Pxx (less SL).
                    # MODEL_INVALID or other errors won't be fixed by lower Pxx — break out.
                    if status not in ('INFEASIBLE',):
                        break
                    continue
                roster, payouts = assign_slabs(
                    activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                    slab_rates_2i, slab_rates_1i, dd_discount
                )
                # Compute profitability stats (Phase A — fast check)
                all_prof = True
                unprof_sum = 0.0
                min_2i_prof = float('inf')
                min_1i_prof = float('inf')
                for v in range(v2):
                    prof = payouts[v] - cost_2i
                    if prof < min_2i_prof:
                        min_2i_prof = prof
                    if prof < -0.5:
                        all_prof = False
                        unprof_sum += -prof
                for v in range(v2, v2 + v1):
                    prof = payouts[v] - cost_1i
                    if prof < min_1i_prof:
                        min_1i_prof = prof
                    if prof < -0.5:
                        all_prof = False
                        unprof_sum += -prof

                print(f'[solver]   Phase A: min_2i=₹{min_2i_prof:.0f}, min_1i=₹{min_1i_prof:.0f}, unprof=₹{unprof_sum:.0f}')

                # Phase B: if Phase A is profitable, run quick CP-SAT slab optimization
                if all_prof:
                    # Phase B per candidate is QUICK — just enough to differentiate candidates.
                    # The winning candidate gets a full polish at the end.
                    remaining = HARD_CAP - (time.time() - search_start)
                    phase_b_time = min(5, max(2, int((remaining - 30) / 3)))  # save 30s for polish
                    print(f'[solver]   Phase A profitable. Running quick Phase B ({phase_b_time}s)...')
                    optimized, opt_status = optimize_slabs_cpsat(
                        activity, v2, days, dd_pairs_per_day, sd_slabs_per_day,
                        slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, phase_b_time
                    )
                    if optimized is not None:
                        opt_roster, opt_payouts = optimized
                        # Recompute stats
                        opt_min_2i = min(opt_payouts[v] - cost_2i for v in range(v2))
                        opt_min_1i = min(opt_payouts[v] - cost_1i for v in range(v2, v2 + v1))
                        print(f'[solver]   Phase B done ({opt_status}): min_2i=₹{opt_min_2i:.0f}, min_1i=₹{opt_min_1i:.0f}')
                        # Use optimized result if better
                        if min(opt_min_2i, opt_min_1i) >= min(min_2i_prof, min_1i_prof):
                            roster = opt_roster
                            payouts = opt_payouts
                            print(f'[solver]   Using Phase B result (better min profit)')
                        else:
                            print(f'[solver]   Phase B worse than Phase A, keeping Phase A')
                    else:
                        print(f'[solver]   Phase B failed ({opt_status}), using Phase A')

                candidate = {
                    'v2': v2, 'v1': v1, 'roster': roster, 'status': status,
                    'pxx_achieved': pxx, 'sl_needed': sl_needed, 'mc': mc, 'payouts': payouts,
                    'all_profitable': all_prof, 'unprofitable_amount': unprof_sum,
                    'activity': activity,  # for polish phase
                }

                # Compute min for tracking
                min_overall_recorded = min(min_2i_prof, min_1i_prof) if (v2 > 0 and v1 > 0) else (min_2i_prof if v2 > 0 else min_1i_prof)
                all_attempts.append({
                    'v2': v2, 'v1': v1, 'total': v2 + v1, 'pxx': pxx,
                    'status': status, 'min_profit': float(min_overall_recorded), 'all_profitable': bool(all_prof),
                })

                if all_prof:
                    # Compute min profit for ranking
                    min_overall = min(min_2i_prof, min_1i_prof)
                    candidate['min_profit'] = min_overall
                    # Check if this improves over current best
                    if profitable_candidates:
                        current_best = min(profitable_candidates, key=lambda c: (c['v2']+c['v1'], -c['min_profit']))
                        current_best_score = (current_best['v2']+current_best['v1'], -current_best['min_profit'])
                        new_score = (v2 + v1, -min_overall)
                        if new_score < current_best_score:
                            iterations_since_improvement = 0
                        else:
                            iterations_since_improvement += 1
                    else:
                        iterations_since_improvement = 0
                    profitable_candidates.append(candidate)
                    print(f'[solver]   PROFITABLE candidate: v2={v2}, v1={v1}, pxx={pxx}, min_profit=₹{min_overall:.0f} (no_improve_streak={iterations_since_improvement})')
                else:
                    # Track best partial
                    if unprof_sum < best_partial_score:
                        best_partial_score = unprof_sum
                        best_partial = candidate
                        print(f'[solver]   new best partial (unprofitable=₹{unprof_sum:.0f})')
                    if profitable_candidates:
                        iterations_since_improvement += 1

    # Done iterating. Pick best candidate.
    if profitable_candidates:
        # Sort by: (1) fewest vendors, (2) highest min profit, (3) tightest spread
        def candidate_score(c):
            spread = max(c['payouts'][:c['v2']]) - min(c['payouts'][:c['v2']]) if c['v2'] > 0 else 0
            return (c['v2'] + c['v1'], -c['min_profit'], spread)
        profitable_candidates.sort(key=candidate_score)
        best = profitable_candidates[0]
        print(f'[solver] Searched {iter_count} iterations. Found {len(profitable_candidates)} profitable candidates.')
        print(f'[solver] Initial pick: v2={best["v2"]}, v1={best["v1"]}, min_profit=₹{best["min_profit"]:.0f}')

        # POLISH PHASE: spend remaining budget squeezing the spread on the winner.
        remaining = HARD_CAP - (time.time() - search_start)
        polish_time = max(5, int(remaining - 3))
        if polish_time >= 5 and 'activity' in best:
            print(f'[solver] Polish phase: {polish_time}s with heavy spread weight (=10)')
            polished_result, polished_status = optimize_slabs_cpsat(
                best['activity'], best['v2'], days, dd_pairs_per_day, sd_slabs_per_day,
                slab_rates_2i, slab_rates_1i, dd_discount, cost_2i, cost_1i, polish_time, spread_weight=10
            )
            if polished_result is not None:
                pol_roster, pol_payouts = polished_result
                # CRITICAL: verify every single vendor is profitable
                pol_v2_profits = [pol_payouts[v] - cost_2i for v in range(best['v2'])]
                pol_v1_profits = [pol_payouts[v] - cost_1i for v in range(best['v2'], best['v2'] + best['v1'])]
                pol_min_2i = min(pol_v2_profits) if pol_v2_profits else 0
                pol_min_1i = min(pol_v1_profits) if pol_v1_profits else 0
                pol_max_2i = max(pol_v2_profits) if pol_v2_profits else 0
                pol_max_1i = max(pol_v1_profits) if pol_v1_profits else 0
                pol_min = min(pol_min_2i, pol_min_1i)
                all_profitable = all(p >= 0 for p in pol_v2_profits) and all(p >= 0 for p in pol_v1_profits)
                if all_profitable and pol_min >= best['min_profit'] - 100:
                    print(f'[solver] Polish done ({polished_status}): 2i spread ₹{pol_max_2i - pol_min_2i:.0f}, 1i spread ₹{pol_max_1i - pol_min_1i:.0f}, min profit ₹{pol_min:.0f}')
                    best['roster'] = pol_roster
                    best['payouts'] = pol_payouts
                    best['min_profit'] = pol_min
                else:
                    if not all_profitable:
                        unprofitable = [i for i, p in enumerate(pol_v2_profits + pol_v1_profits) if p < 0]
                        print(f'[solver] Polish made vendors unprofitable: {len(unprofitable)} vendors below ₹0. Keeping pre-polish.')
                    else:
                        print(f'[solver] Polish reduced min profit. Keeping pre-polish.')
            else:
                print(f'[solver] Polish failed ({polished_status}), keeping pre-polish')
        best['all_attempts'] = all_attempts
        return best
    # Loop done without fully profitable. Return best partial if any.
    if best_partial:
        print(f'[solver] Exhausted search. Returning best partial: v2={best_partial["v2"]}, v1={best_partial["v1"]}, unprofitable_sum=₹{best_partial_score:.0f}')
        best_partial['all_attempts'] = all_attempts
        return best_partial
    return None


def run_solver(params):
    total_sites = int(params['total_sites'])
    days = int(params.get('days', 30))
    peak_ratio = float(params['peak_ratio'])
    elig_pct = float(params['elig_pct'])
    sl2_rate = float(params['sl2_rate'])
    sl1_rate = float(params['sl1_rate'])
    slab_rates_2i = [float(x) for x in params['slab_rates']]
    # If slab_rates_1i not provided, fall back to same as 2i (backward compatibility)
    slab_rates_1i = [float(x) for x in params.get('slab_rates_1i', params['slab_rates'])]
    slab_mix = [float(x) for x in params['slab_mix']]
    dd_elig_slabs = [bool(x) for x in params['dd_elig_slabs']]
    cost_2i = float(params['cost_2i'])
    cost_1i = float(params['cost_1i'])
    dd_discount = float(params['dd_discount'])
    baseline_per_site = float(params.get('baseline_per_site', 10572))
    target_pxx = float(params.get('target_pxx', 0.75))
    time_limit = int(params.get('time_limit_sec', 30))
    max_working_days = int(params.get('max_working_days', 26))

    daily = bell_curve_demand(total_sites, days, peak_ratio)
    dd_pairs, sd_slabs = compute_daily_demand(daily, slab_mix, dd_elig_slabs, elig_pct)
    pair_count = [len(p) for p in dd_pairs]
    sd_count = [sum(s) for s in sd_slabs]
    peak_day = daily.index(max(daily))

    print(f'[solver] daily sum={sum(daily)} expected={total_sites}, peak={peak_day}, max_work_days={max_working_days}')
    print(f'[solver] 2i rates: {slab_rates_2i}')
    print(f'[solver] 1i rates: {slab_rates_1i}')

    result = find_optimal_fast(daily, dd_pairs, sd_slabs, pair_count, sd_count,
                               peak_day, sl2_rate, sl1_rate, target_pxx, slab_rates_2i, slab_rates_1i,
                               dd_discount, cost_2i, cost_1i, time_limit, max_working_days)

    if result is None:
        return {
            'ok': False,
            'reason': 'No profitable solution found AND no feasible schedule found. At this slip rate + cost combination, the math doesn\'t work. Try: lowering slip%, raising DD discount, lowering 2i fixed cost, or lowering peak ratio.',
            'daily': daily,
            'pair_count_per_day': pair_count,
            'sd_count_per_day': sd_count,
            'peak_day': peak_day,
            'total_slips': round(sum([p*2 for p in pair_count]) * sl2_rate + sum(sd_count) * sl1_rate),
        }

    v2 = result['v2']
    v1 = result['v1']
    total_v = v2 + v1
    roster = result['roster']

    vendors = []
    for vi in range(total_v):
        is_v2 = vi < v2
        dd_d = sd_d = sl_d = idle_d = 0
        payout = 0
        # 2i uses 2i rates; 1i uses 1i rates (1i can't do DD, only SD)
        rates = slab_rates_2i if is_v2 else slab_rates_1i
        for d in range(days):
            cell = roster[vi][d]
            if cell[0] == 'DD':
                s1, s2 = cell[1]
                # DD always uses 2i rates with discount
                r1, r2 = slab_rates_2i[s1], slab_rates_2i[s2]
                payout += max(r1, r2) + min(r1, r2) * dd_discount
                dd_d += 1
            elif cell[0] == 'SD':
                payout += rates[cell[1]]
                sd_d += 1
            elif cell[0] == 'SL':
                sl_d += 1
            else:
                idle_d += 1
        fixed = cost_2i if is_v2 else cost_1i
        profit = payout - fixed
        vendors.append({
            'name': f'V{vi+1:02d}',
            'type': '2-Install' if is_v2 else '1-Install',
            'dd_days': dd_d, 'sd_sites': sd_d, 'sl_days': sl_d, 'idle_days': idle_d,
            'sites': dd_d * 2 + sd_d, 'fixed_cost': fixed,
            'payout': round(payout), 'profit': round(profit),
        })

    slab_labels = ['S1', 'S2', 'S3', 'S4']
    roster_out = []
    for vi in range(total_v):
        row = []
        for d in range(days):
            cell = roster[vi][d]
            if cell[0] == 'DD':
                s1, s2 = cell[1]
                row.append({'type': 'DD', 'label': f'{slab_labels[s1]}+{slab_labels[s2]}'})
            elif cell[0] == 'SD':
                row.append({'type': 'SD', 'label': slab_labels[cell[1]]})
            elif cell[0] == 'SL':
                row.append({'type': 'SL', 'label': ''})
            else:
                row.append({'type': 'idle', 'label': ''})
        roster_out.append(row)

    total_payout = sum(v['payout'] for v in vendors)
    total_cost = sum(v['fixed_cost'] for v in vendors)
    total_profit = sum(v['profit'] for v in vendors)
    baseline = total_sites * baseline_per_site
    savings_pct = 100 * (baseline - total_payout) / baseline if baseline > 0 else 0

    v2_list = [v for v in vendors if v['type'] == '2-Install']
    v1_list = [v for v in vendors if v['type'] == '1-Install']
    avg_2i = sum(v['profit'] for v in v2_list) / len(v2_list) if v2_list else 0
    avg_1i = sum(v['profit'] for v in v1_list) / len(v1_list) if v1_list else 0
    min_2i = min((v['profit'] for v in v2_list), default=0)
    min_1i = min((v['profit'] for v in v1_list), default=0)

    return {
        'ok': True,
        'all_profitable': result.get('all_profitable', True),
        'unprofitable_amount': round(result.get('unprofitable_amount', 0)),
        'status': result['status'],
        'v2': v2, 'v1': v1, 'total_v': total_v,
        'daily': daily,
        'pair_count_per_day': pair_count,
        'sd_count_per_day': sd_count,
        'sl_needed_by_day': result['sl_needed'],
        'peak_day': peak_day,
        'roster': roster_out,
        'vendors': vendors,
        'total_payout': total_payout,
        'total_cost': total_cost,
        'total_profit': total_profit,
        'savings_pct': round(savings_pct, 1),
        'avg_2i': round(avg_2i),
        'avg_1i': round(avg_1i),
        'min_2i': round(min_2i),
        'min_1i': round(min_1i),
        'total_slips': result['mc']['total_slips'],
        'pxx_target': target_pxx,
        'pxx_achieved': result['pxx_achieved'],
        'all_attempts': result.get('all_attempts', []),
    }


# ============================================================
# PDF REPORT GENERATION
# ============================================================

BRAND_COLOR = colors.HexColor('#131ac3') if REPORTLAB_OK else None
BRAND_LIGHT = colors.HexColor('#e3e4f8') if REPORTLAB_OK else None
ACCENT_GREEN = colors.HexColor('#187a3b') if REPORTLAB_OK else None
ACCENT_RED = colors.HexColor('#c00000') if REPORTLAB_OK else None
GREY_LIGHT = colors.HexColor('#f5f5f7') if REPORTLAB_OK else None


def build_pdf_report(data, city_name):
    """Generate a CFO-grade PDF report. Returns bytes."""
    if not REPORTLAB_OK:
        raise RuntimeError('reportlab not installed. Run: pip3 install reportlab --break-system-packages')

    # ---- Page setup: A4 portrait for cover/summary, landscape for big tables ----
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm,
        topMargin=14 * mm, bottomMargin=16 * mm,
        title=f'SolarSquare Vendor Plan — {city_name}',
        author='SolarSquare Energy',
    )

    # ---- Colors ----
    NAVY = colors.HexColor('#131ac3')        # primary brand
    NAVY_DEEP = colors.HexColor('#0a1080')   # darker
    LAVENDER = colors.HexColor('#eef0ff')    # tint for backgrounds
    LAVENDER_DARK = colors.HexColor('#dcdefb')
    INK = colors.HexColor('#0f172a')         # near-black text
    SLATE = colors.HexColor('#475569')       # secondary text
    MUTED = colors.HexColor('#94a3b8')       # tertiary
    HAIRLINE = colors.HexColor('#e2e8f0')
    SUCCESS = colors.HexColor('#15803d')
    SUCCESS_BG = colors.HexColor('#dcfce7')
    DANGER = colors.HexColor('#b91c1c')
    DANGER_BG = colors.HexColor('#fee2e2')
    AMBER = colors.HexColor('#d97706')
    AMBER_BG = colors.HexColor('#fef3c7')
    PAPER = colors.HexColor('#ffffff')
    SOFT = colors.HexColor('#f8fafc')

    # ---- Styles ----
    def P(text, size=10, color=INK, bold=False, align=TA_LEFT, leading=None, font=None):
        fn = font or (FONT_BOLD if bold else FONT_REGULAR)
        return Paragraph(text, ParagraphStyle(
            f's{id(text) & 0xffff}',
            fontName=fn, fontSize=size, leading=leading or (size * 1.3),
            textColor=color, alignment=align, spaceAfter=0, spaceBefore=0,
        ))

    H1 = ParagraphStyle('H1', fontName=FONT_BOLD, fontSize=24, leading=28,
                        textColor=NAVY, spaceAfter=2)
    H2 = ParagraphStyle('H2', fontName=FONT_BOLD, fontSize=14, leading=18,
                        textColor=NAVY, spaceBefore=14, spaceAfter=6)
    SECTION_LABEL = ParagraphStyle('SECT', fontName=FONT_BOLD, fontSize=8, leading=10,
                                   textColor=NAVY, spaceAfter=0,
                                   alignment=TA_LEFT)
    SUBTITLE = ParagraphStyle('SUB', fontName=FONT_REGULAR, fontSize=10,
                              leading=14, textColor=SLATE, spaceAfter=12)
    BODY = ParagraphStyle('B', fontName=FONT_REGULAR, fontSize=10,
                          leading=14, textColor=INK)
    TINY = ParagraphStyle('T', fontName=FONT_REGULAR, fontSize=7,
                          leading=9, textColor=MUTED)

    # ---- Helpers ----
    def fmt_inr(v):
        v = round(v)
        sign = '-' if v < 0 else ''
        v = abs(v)
        if v >= 1e7: return f'{sign}₹{v/1e7:.2f} Cr'
        if v >= 1e5: return f'{sign}₹{v/1e5:.2f} L'
        if v >= 1e3: return f'{sign}₹{v/1e3:.1f}K'
        return f'{sign}₹{v:.0f}'

    def fmt_inr_full(v):
        v = round(v)
        sign = '-' if v < 0 else ''
        v = abs(v)
        # Indian number system grouping
        s = f'{v:,}'.replace(',', '_')  # use _ as temp separator
        # Convert to Indian: last 3, then groups of 2
        if v >= 1000:
            s = str(v)
            last_three = s[-3:]
            rest = s[:-3]
            parts = []
            while len(rest) > 2:
                parts.append(rest[-2:])
                rest = rest[:-2]
            if rest:
                parts.append(rest)
            return f'{sign}₹{",".join(reversed(parts))},{last_three}'
        return f'{sign}₹{v}'

    def pct(arr, p):
        if not arr: return None
        s = sorted(arr)
        if p <= 0: return s[0]
        if p >= 100: return s[-1]
        idx = (p / 100.0) * (len(s) - 1)
        lo = int(idx); hi = lo + 1 if lo + 1 < len(s) else lo
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)

    now = datetime.datetime.now()
    date_str = now.strftime('%d %B %Y')
    time_str = now.strftime('%H:%M')

    story = []

    # ============================================================
    # PAGE 1: COVER + EXECUTIVE SUMMARY
    # ============================================================

    # ---- Brand banner ----
    banner_tbl = Table([
        [P('S', size=20, color=PAPER, bold=True),
         P('SolarSquare', size=14, color=PAPER, bold=True)]
    ], colWidths=[10*mm, 60*mm])
    banner_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (0,0), 6),
        ('LEFTPADDING', (1,0), (1,0), 4),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    # Right side: date
    header_row = Table([[banner_tbl, P(f'{date_str}  ·  {time_str}', size=9, color=SLATE, align=TA_RIGHT)]],
                       colWidths=[70*mm, 108*mm])
    header_row.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, HAIRLINE),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(header_row)
    story.append(Spacer(1, 18))

    # ---- Title block ----
    story.append(Paragraph('Vendor Plan', H1))
    story.append(Paragraph(f'<font color="#475569">{city_name}  ·  Monthly Optimisation Report</font>',
                           ParagraphStyle('TT', fontName=FONT_REGULAR, fontSize=13, leading=16,
                                          textColor=SLATE, spaceAfter=4)))
    story.append(Spacer(1, 4))
    # Status pill
    status_color = SUCCESS if data.get('status') == 'OPTIMAL' else AMBER
    status_bg = SUCCESS_BG if data.get('status') == 'OPTIMAL' else AMBER_BG
    pill = Table([[P(f'  {data["status"]}  ', size=8, color=status_color, bold=True),
                   P(f'  Pxx achieved: P{round(data["pxx_achieved"]*100)}  ', size=8, color=NAVY, bold=True)]],
                 colWidths=[28*mm, 38*mm])
    pill.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), status_bg),
        ('BACKGROUND', (1,0), (1,0), LAVENDER),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pill)
    story.append(Spacer(1, 22))

    # ---- Hero KPI cards (2x3 grid) ----
    story.append(Paragraph('AT A GLANCE', SECTION_LABEL))
    story.append(Spacer(1, 4))

    total_v = data['v2'] + data['v1']

    def kpi_card(label, value, accent_color=NAVY, sub=None):
        rows = [
            [P(label, size=8, color=MUTED, bold=True)],
            [P(value, size=22, color=accent_color, bold=True, leading=24)],
        ]
        if sub:
            rows.append([P(sub, size=8, color=SLATE)])
        t = Table(rows, colWidths=[57*mm])
        style = [
            ('BACKGROUND', (0,0), (-1,-1), PAPER),
            ('BOX', (0,0), (-1,-1), 0.5, HAIRLINE),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (0,0), 10),
            ('TOPPADDING', (0,1), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-2), 0),
            ('BOTTOMPADDING', (0,-1), (-1,-1), 10),
        ]
        t.setStyle(TableStyle(style))
        return t

    # Row 1
    row1 = Table([[
        kpi_card('TOTAL VENDORS', str(total_v), accent_color=NAVY,
                 sub=f'{data["v2"]} two-install · {data["v1"]} one-install'),
        kpi_card('MONTHLY PAYOUT', fmt_inr(data['total_payout']), accent_color=NAVY,
                 sub=fmt_inr_full(data['total_payout'])),
        kpi_card('SSE SAVINGS', f'{data["savings_pct"]}%', accent_color=SUCCESS,
                 sub=f'vs baseline {fmt_inr(data["total_cost"] / (1 - data["savings_pct"]/100) if data["savings_pct"] < 100 else data["total_payout"])}'),
    ]], colWidths=[58*mm, 58*mm, 58*mm])
    row1.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(row1)
    story.append(Spacer(1, 8))

    # Row 2
    row2 = Table([[
        kpi_card('SITES SERVED', str(sum(data['daily'])), accent_color=NAVY,
                 sub=f'Peak day: Day {data["peak_day"]+1} ({max(data["daily"])} sites)'),
        kpi_card('SLIPS COVERED', str(data['total_slips']), accent_color=NAVY,
                 sub='Deterministic monthly slip count'),
        kpi_card('PROFIT MARGIN', f'{round(100 * data["total_profit"] / data["total_payout"] if data["total_payout"] else 0, 1)}%',
                 accent_color=SUCCESS if data['total_profit'] >= 0 else DANGER,
                 sub=f'Net: {fmt_inr(data["total_profit"])}'),
    ]], colWidths=[58*mm, 58*mm, 58*mm])
    row2.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(row2)

    # ---- Profit distribution table ----
    story.append(Spacer(1, 18))
    story.append(Paragraph('PROFIT DISTRIBUTION', SECTION_LABEL))
    story.append(Spacer(1, 6))

    profits_2i = [v['profit'] for v in data['vendors'] if v['type'] == '2-Install']
    profits_1i = [v['profit'] for v in data['vendors'] if v['type'] == '1-Install']
    avg_2i = sum(profits_2i) / len(profits_2i) if profits_2i else 0
    avg_1i = sum(profits_1i) / len(profits_1i) if profits_1i else 0

    def prof_cell(v):
        if v is None:
            return P('—', size=10, color=MUTED, align=TA_RIGHT)
        color = SUCCESS if v >= 0 else DANGER
        return P(fmt_inr(v), size=10, color=color, bold=True, align=TA_RIGHT)

    pct_data = [
        [P('Percentile', size=9, color=PAPER, bold=True),
         P('2-Install', size=9, color=PAPER, bold=True, align=TA_RIGHT),
         P('1-Install', size=9, color=PAPER, bold=True, align=TA_RIGHT)],
    ]
    for label, p in [('P0 (Min)', 0), ('P25', 25), ('P50 (Median)', 50), ('P75', 75), ('P90', 90), ('P100 (Max)', 100)]:
        pct_data.append([
            P(label, size=10, color=INK),
            prof_cell(pct(profits_2i, p)),
            prof_cell(pct(profits_1i, p)),
        ])
    pct_data.append([
        P('Average', size=10, color=NAVY, bold=True),
        P(fmt_inr(avg_2i), size=11, color=NAVY, bold=True, align=TA_RIGHT),
        P(fmt_inr(avg_1i), size=11, color=NAVY, bold=True, align=TA_RIGHT),
    ])

    pct_table = Table(pct_data, colWidths=[80*mm, 49*mm, 49*mm])
    pct_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [PAPER, SOFT]),
        ('BACKGROUND', (0,-1), (-1,-1), LAVENDER),
        ('LINEABOVE', (0,-1), (-1,-1), 1.5, NAVY),
        ('LINEBELOW', (0,0), (-1,0), 0, PAPER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 0.5, HAIRLINE),
    ]))
    story.append(pct_table)

    # ---- Footer note ----
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        f'Generated {date_str} at {time_str} · SolarSquare 2-Install-A-Day Vendor Optimizer',
        TINY
    ))

    # ============================================================
    # PAGE 2: PER-VENDOR P&L
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph('Per-Vendor P&amp;L', H2))
    story.append(Paragraph(f'Detailed financials for all {total_v} vendors in {city_name}.', SUBTITLE))

    # Build vendor table
    header_row = [
        P('VENDOR', size=8, color=PAPER, bold=True),
        P('TYPE', size=8, color=PAPER, bold=True),
        P('DD', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('SD', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('SL', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('IDLE', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('SITES', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('FIXED', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('PAYOUT', size=8, color=PAPER, bold=True, align=TA_RIGHT),
        P('PROFIT', size=8, color=PAPER, bold=True, align=TA_RIGHT),
    ]
    pnl_rows = [header_row]
    for v in data['vendors']:
        prof_color = SUCCESS if v['profit'] >= 0 else DANGER
        pnl_rows.append([
            P(v['name'], size=9, color=INK, bold=True),
            P(v['type'].replace('-Install', '-Inst.'), size=9, color=SLATE),
            P(str(v['dd_days']), size=9, color=INK, align=TA_RIGHT),
            P(str(v['sd_sites']), size=9, color=INK, align=TA_RIGHT),
            P(str(v['sl_days']), size=9, color=INK, align=TA_RIGHT),
            P(str(v['idle_days']), size=9, color=MUTED, align=TA_RIGHT),
            P(str(v['sites']), size=9, color=INK, bold=True, align=TA_RIGHT),
            P(fmt_inr(v['fixed_cost']), size=9, color=SLATE, align=TA_RIGHT),
            P(fmt_inr(v['payout']), size=9, color=INK, align=TA_RIGHT),
            P(fmt_inr(v['profit']), size=10, color=prof_color, bold=True, align=TA_RIGHT),
        ])
    # Totals
    pnl_rows.append([
        P('TOTAL', size=10, color=NAVY, bold=True),
        P(f'{data["v2"]+data["v1"]} vendors', size=9, color=NAVY),
        P(str(sum(v['dd_days'] for v in data['vendors'])), size=10, color=NAVY, bold=True, align=TA_RIGHT),
        P(str(sum(v['sd_sites'] for v in data['vendors'])), size=10, color=NAVY, bold=True, align=TA_RIGHT),
        P(str(sum(v['sl_days'] for v in data['vendors'])), size=10, color=NAVY, bold=True, align=TA_RIGHT),
        P(str(sum(v['idle_days'] for v in data['vendors'])), size=10, color=MUTED, bold=True, align=TA_RIGHT),
        P(str(sum(v['sites'] for v in data['vendors'])), size=10, color=NAVY, bold=True, align=TA_RIGHT),
        P(fmt_inr(data['total_cost']), size=9, color=NAVY, bold=True, align=TA_RIGHT),
        P(fmt_inr(data['total_payout']), size=10, color=NAVY, bold=True, align=TA_RIGHT),
        P(fmt_inr(data['total_profit']), size=11, color=SUCCESS if data['total_profit'] >= 0 else DANGER, bold=True, align=TA_RIGHT),
    ])

    page_w = 178 * mm  # A4 portrait minus margins
    col_widths_pnl = [16, 18, 10, 10, 10, 10, 12, 24, 28, 30]
    total_units = sum(col_widths_pnl)
    col_widths_pnl = [w / total_units * page_w for w in col_widths_pnl]

    pnl_table = Table(pnl_rows, colWidths=col_widths_pnl, repeatRows=1)
    pnl_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [PAPER, SOFT]),
        ('BACKGROUND', (0,-1), (-1,-1), LAVENDER),
        ('LINEABOVE', (0,-1), (-1,-1), 1.5, NAVY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 0.5, HAIRLINE),
        ('LINEBELOW', (0,1), (-1,-2), 0.25, HAIRLINE),
    ]))
    story.append(pnl_table)

    # Glossary
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        '<b>Legend:</b>  DD = double install day (2 sites)  ·  SD = single install day  ·  SL = slip recovery  ·  Fixed = monthly fixed cost  ·  Profit = Payout − Fixed',
        TINY
    ))

    # ============================================================
    # PAGE 3 (landscape): VENDOR ROSTER CALENDAR
    # ============================================================
    # Switch to landscape — use NextPageTemplate approach. Simpler: just continue with page break.
    # For visual fit we'll render the calendar compactly.
    story.append(PageBreak())
    story.append(Paragraph('Daily Demand', H2))
    story.append(Paragraph('Site demand bell curve across the month. Peak day highlighted.', SUBTITLE))

    daily = data['daily']
    peak_day = data['peak_day']
    max_d = max(daily) if daily else 1

    # Render as horizontal bar table with subtle background gradient
    n_days = len(daily)
    chart_data = []
    # Day labels row
    chart_data.append([P(f'D{i+1}', size=7, color=MUTED, align=TA_CENTER) for i in range(n_days)])
    # Bars row — use background color intensity to represent value
    bar_row = []
    for i, v in enumerate(daily):
        is_peak = (i == peak_day)
        col = AMBER if is_peak else NAVY
        bar_row.append(P(f'<b>{v}</b>', size=9, color=col, align=TA_CENTER, bold=True))
    chart_data.append(bar_row)

    col_w_chart = (178 / n_days) * mm
    chart_table = Table(chart_data, colWidths=[col_w_chart]*n_days, rowHeights=[5*mm, 8*mm])
    chart_style = [
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LINEBELOW', (0,0), (-1,0), 0.25, HAIRLINE),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]
    # Background tint for bars proportional to value
    for i, v in enumerate(daily):
        intensity = v / max_d if max_d else 0
        if i == peak_day:
            chart_style.append(('BACKGROUND', (i, 1), (i, 1), AMBER_BG))
        else:
            # Light lavender tint scaled by demand
            shade_pct = 0.15 + intensity * 0.45
            rgb_val = int(255 - shade_pct * 60)
            tint = colors.HexColor(f'#{rgb_val:02x}{rgb_val:02x}fb')
            chart_style.append(('BACKGROUND', (i, 1), (i, 1), tint))
    chart_table.setStyle(TableStyle(chart_style))
    story.append(chart_table)

    story.append(Spacer(1, 6))
    # Demand summary stats
    avg_daily = sum(daily) / len(daily)
    peak_to_avg = max(daily) / avg_daily if avg_daily else 1
    total_dd_pairs = sum(data['pair_count_per_day'])
    total_sd_sites = sum(data['sd_count_per_day'])
    summary_data = [
        [P('Total sites', size=8, color=MUTED, bold=True),
         P('Peak day', size=8, color=MUTED, bold=True),
         P('Average day', size=8, color=MUTED, bold=True),
         P('Peak/Avg ratio', size=8, color=MUTED, bold=True),
         P('DD pairs', size=8, color=MUTED, bold=True),
         P('SD sites', size=8, color=MUTED, bold=True)],
        [P(str(sum(daily)), size=13, color=NAVY, bold=True),
         P(f'{max(daily)} (Day {peak_day+1})', size=13, color=AMBER, bold=True),
         P(f'{avg_daily:.1f}', size=13, color=NAVY, bold=True),
         P(f'{peak_to_avg:.2f}x', size=13, color=NAVY, bold=True),
         P(str(total_dd_pairs), size=13, color=NAVY, bold=True),
         P(str(total_sd_sites), size=13, color=NAVY, bold=True)],
    ]
    summary_t = Table(summary_data, colWidths=[178/6*mm]*6, rowHeights=[6*mm, 10*mm])
    summary_t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0,0), (-1,-1), SOFT),
        ('BOX', (0,0), (-1,-1), 0.5, HAIRLINE),
    ]))
    story.append(summary_t)

    # ============================================================
    # PAGE 4: VENDOR ROSTER CALENDAR
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph('Vendor Roster Calendar', H2))
    story.append(Paragraph('Daily activity for each vendor across the month.', SUBTITLE))

    # Legend
    legend_data = [[
        P('●', size=10, color=SUCCESS, bold=True), P('DD (double install)', size=8, color=SLATE),
        P('●', size=10, color=NAVY, bold=True), P('SD (single install)', size=8, color=SLATE),
        P('●', size=10, color=DANGER, bold=True), P('SL (slip recovery)', size=8, color=SLATE),
        P('●', size=10, color=MUTED, bold=True), P('Idle', size=8, color=SLATE),
    ]]
    legend_t = Table(legend_data, colWidths=[5*mm, 32*mm, 5*mm, 32*mm, 5*mm, 32*mm, 5*mm, 20*mm])
    legend_t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(legend_t)
    story.append(Spacer(1, 8))

    roster = data['roster']
    n_vendors = len(roster)
    days = len(daily)

    # Header
    cal_header = [P('DAY', size=7, color=PAPER, bold=True, align=TA_CENTER)]
    for v in data['vendors']:
        cal_header.append(P(v['name'], size=7, color=PAPER, bold=True, align=TA_CENTER))
    cal_rows = [cal_header]
    cstyle = [
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 1),
        ('RIGHTPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BOX', (0,0), (-1,-1), 0.5, HAIRLINE),
        ('LINEBELOW', (0,0), (-1,0), 0.5, NAVY),
    ]

    for d in range(days):
        is_peak_row = (d == peak_day)
        row = [P(f'D{d+1}', size=7, color=NAVY if is_peak_row else SLATE, bold=is_peak_row, align=TA_CENTER)]
        for vi in range(n_vendors):
            cell = roster[vi][d]
            kind = cell.get('type', 'idle')
            label = cell.get('label', '') or ''
            if kind == 'DD':
                row.append(P(f'<b>DD</b><br/>{label}', size=6, color=PAPER, bold=True, align=TA_CENTER, leading=7))
                cstyle.append(('BACKGROUND', (vi+1, d+1), (vi+1, d+1), SUCCESS))
            elif kind == 'SD':
                row.append(P(f'<b>SD</b><br/>{label}', size=6, color=INK, align=TA_CENTER, leading=7))
                cstyle.append(('BACKGROUND', (vi+1, d+1), (vi+1, d+1), LAVENDER))
            elif kind == 'SL':
                row.append(P('<b>SL</b>', size=7, color=PAPER, bold=True, align=TA_CENTER))
                cstyle.append(('BACKGROUND', (vi+1, d+1), (vi+1, d+1), DANGER))
            else:
                row.append(P('—', size=7, color=MUTED, align=TA_CENTER))
        cal_rows.append(row)
        # Peak day row highlight on the day cell
        if is_peak_row:
            cstyle.append(('BACKGROUND', (0, d+1), (0, d+1), AMBER_BG))

    # Calendar fits in landscape — but we're in portrait. Let it overflow via repeat split.
    page_w = 178 * mm
    day_col = 11 * mm
    vendor_col = (page_w - day_col) / max(1, n_vendors)
    col_widths_cal = [day_col] + [vendor_col] * n_vendors
    cal_table = Table(cal_rows, colWidths=col_widths_cal, repeatRows=1)
    cal_table.setStyle(TableStyle(cstyle))
    story.append(cal_table)

    # ---- Build ----
    def add_page_decoration(canvas, doc_):
        canvas.saveState()
        # Footer line
        canvas.setStrokeColor(HAIRLINE)
        canvas.setLineWidth(0.5)
        canvas.line(16*mm, 12*mm, A4[0] - 16*mm, 12*mm)
        # Footer text
        canvas.setFont(FONT_REGULAR, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(16*mm, 8*mm, f'SolarSquare Vendor Plan — {city_name}')
        canvas.drawRightString(A4[0] - 16*mm, 8*mm, f'Page {doc_.page}  ·  {date_str}')
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_decoration, onLaterPages=add_page_decoration)
    return buf.getvalue()



@app.route('/report', methods=['POST'])
def report_route():
    if not REPORTLAB_OK:
        return jsonify({'ok': False, 'reason': 'reportlab not installed on server. Run: pip3 install reportlab --break-system-packages'}), 500
    try:
        body = request.get_json()
        city = (body.get('city') or '').strip() or 'Unspecified City'
        data = body.get('data')
        if not data or not data.get('ok'):
            return jsonify({'ok': False, 'reason': 'no solution data provided'}), 400
        pdf_bytes = build_pdf_report(data, city)
        safe_city = ''.join(c for c in city if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_') or 'city'
        date_str = datetime.datetime.now().strftime('%Y%m%d')
        filename = f'SolarSquare_VendorPlan_{safe_city}_{date_str}.pdf'
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        tb = tb_mod.format_exc()
        print(f'[/report] ERROR: {tb}')
        return jsonify({'ok': False, 'reason': str(e), 'traceback': tb}), 500


@app.route('/')
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/health')
def health():
    return jsonify({'ortools_loaded': ORTOOLS_OK, 'ortools_error': ORTOOLS_ERR})


@app.route('/solve', methods=['POST'])
def solve_route():
    if not ORTOOLS_OK:
        return jsonify({
            'ok': False,
            'reason': f'OR-Tools missing: {ORTOOLS_ERR}'
        }), 200
    try:
        params = request.get_json(force=True, silent=False)
        if params is None:
            return jsonify({'ok': False, 'reason': 'No JSON body'}), 200
        print(f'[/solve] received request')
        result = run_solver(params)
        return jsonify(result), 200
    except Exception as e:
        tb = tb_mod.format_exc()
        print('=' * 60)
        print('ERROR in /solve:')
        print(tb)
        print('=' * 60)
        return jsonify({
            'ok': False, 'reason': f'{type(e).__name__}: {e}', 'traceback': tb,
        }), 200


@app.errorhandler(404)
def handle_404(e):
    return jsonify({'ok': False, 'reason': f'404: {request.path}'}), 200


@app.errorhandler(500)
def handle_500(e):
    return jsonify({'ok': False, 'reason': f'500: {e}', 'traceback': tb_mod.format_exc()}), 200


if __name__ == '__main__':
    print('=' * 60)
    print('SolarSquare CP-SAT Solver (v34 — FAST)')
    print(f'OR-Tools loaded: {ORTOOLS_OK}')
    if not ORTOOLS_OK:
        print(f'  Error: {ORTOOLS_ERR}')
    print('Open http://localhost:5000 in browser')
    print('=' * 60)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
