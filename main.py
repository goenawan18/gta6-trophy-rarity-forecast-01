"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads every <id>.in case in the input directory and writes one <id>.out forecast
per case: five quantiles of each trophy's Exophase earn rate at the case target.

Model — refresh-aware saturating logit trend, written fresh for this entry:
  * Exophase's page is a snapshot: it stays frozen between refreshes and jumps at
    each refresh, when newly tracked players dilute most rates (the practice
    histories are flat for hours, then one refresh took players 638 -> 3868 and
    every rate but the pinned 100% one fell). So the signal lives in the distinct
    page states, not in the raw 10-minute reads: consecutive identical reads are
    collapsed to one state each.
  * Per trophy, a saturating curve in log-time: rate(x) = pi * sigmoid(a + b*x)
    with x = ln(1 + hours since release), i.e. logit(rate/pi) is linear in x.
    pi (the eventual ceiling) is grid-searched per trophy with a parsimony
    rule (the smallest ceiling whose fit is within 10% of the best), and the
    slope b is shrunk toward the tier's pooled slope (platinum / gold /
    silver / bronze). A straight logit line over the whole window would
    overshoot badly — the growth curve is concave in log-time, so its chord
    slope is far above its forward slope — while the saturating curve flattens
    where the data say it should. The fit already nets progression against
    dilution, because both show up in the observed rates.
  * A trophy whose history is flat (one state, no trend to fit) gets a dilution
    prior instead of persistence: a refresh before the target is expected, and a
    refresh pulls rates down by ~0.30 logit each, so the median is shifted down
    by 0.30 x the expected number of refreshes before the target.
  * The spread is honest, not assumed: rolling-origin backtests inside the
    history (fit on states up to k, forecast state k+1) pool the logit errors
    into a residual scale, which is then widened by the fit's extrapolation
    leverage at the target and floored at 0.30 logit (one refresh can move a
    rate that far). Quantiles are the median plus the normal z of each level
    times that spread, on the logit scale.
  * A trophy pinned at ~100% stays ~100% (it survived a 6x dilution untouched);
    a trophy at ~0% gets a low median with a wide upward spread.

Deterministic: no randomness, no wall clock, no network; iteration follows the
case's own order everywhere. Standard library only. Runs in well under 8 s.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]
LO, HI = 0.01, 99.99

# --- model constants (see the module docstring for why each exists) ---
RECENCY_HALF_LIFE_H = 36.0   # recency weight half-life over distinct states
SHRINK_PRIOR_W = 2.0         # prior weight (in effective states) of tier slope
SLOPE_CAP = 3.0              # |logit slope| cap, per log-time unit
REFRESH_DRIFT = 0.30         # logit fall per expected refresh (dilution prior)
REFRESH_PRIOR_N = 1.0        # pseudo-refreshes added when estimating refresh rate
REFRESH_PRIOR_H = 5.0        # pseudo-hours added (prior: ~1 refresh per 5 h)
SIGMA_FLOOR = 0.30           # minimum logit spread (one refresh moves ~this far)
SIGMA_CAP = 1.50             # maximum logit spread
SIGMA_FLAT = 0.60            # spread when no trend/backtest exists (flat history)
SIGMA_POOL_DEFAULT = 0.45    # residual scale fallback with too few backtests
LEV_CAP = 4.0                # cap on extrapolation-leverage variance factor
PINNED_RATE = 99.5           # a rate at/above this is treated as pinned at ~100
FLOOR_RATE = 0.05            # a rate at/below this is treated as ~unearned yet
PI_MULTS = (1.05, 1.2, 1.4, 1.7, 2.2, 3.0, 4.5, 7.0, 12.0)
TIER_PI_CAP = {"platinum": 15.0, "gold": 50.0, "silver": 90.0, "bronze": 99.0}


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def parse_time(text):
    """ISO-8601 timestamp (or bare date) -> aware UTC datetime."""
    text = str(text).strip()
    if "T" not in text:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(HI, max(LO, x))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg)) if lg < 700 else 100.0
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (the contest's rounding rule)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def median_of(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


# --------------------------------------------------------------------------
# history -> distinct page states
# --------------------------------------------------------------------------

def distinct_states(case):
    """Collapse consecutive identical reads to one state each.

    Returns [(time_dt, players_tracked, rates_tuple)] in history order.
    The page only changes at refreshes, so one entry per distinct snapshot
    is the whole signal; the raw read cadence carries no information.
    """
    states = []
    prev_key = None
    for read in case.get("history") or []:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key == prev_key:
            continue
        prev_key = key
        states.append((parse_time(read["at"]), read.get("players_tracked"), rates))
    return states


def release_time(case):
    """Midnight UTC of the game's release date, else the first read's time."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            return parse_time(raw)
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return parse_time(history[0]["at"])
    return parse_time(case.get("at"))


def hours_since(t, t0):
    return max(0.0, (t - t0).total_seconds() / 3600.0)


def log_time(t, t0):
    """ln(1 + hours since release): the clock the growth curve is linear in."""
    return math.log1p(hours_since(t, t0))


# --------------------------------------------------------------------------
# per-trophy weighted line fit of logit(rate) on log-time
# --------------------------------------------------------------------------

def wls(points):
    """Weighted least squares y ~ a + b*x over [(x, y, w)].

    Returns (ybar, b, xbar, sxx, n_eff, resid_sd, n) or None when unusable.
    Two passes, no per-point function calls: the fit runs thousands of
    times inside the ceiling grid search and the rolling backtest.
    """
    if len(points) < 2:
        return None
    sw = sx = sy = sw2 = 0.0
    for x, y, w in points:
        sw += w
        sx += w * x
        sy += w * y
        sw2 += w * w
    if sw <= 0:
        return None
    xbar = sx / sw
    ybar = sy / sw
    sxx = 0.0
    sxy = 0.0
    for x, y, w in points:
        dx = x - xbar
        sxx += w * dx * dx
        sxy += w * dx * (y - ybar)
    if sxx <= 1e-12:
        return None
    b = sxy / sxx
    n_eff = sw * sw / sw2 if sw2 > 0 else 0.0
    resid_sd = None
    if len(points) >= 3:
        sse = 0.0
        for x, y, w in points:
            d = y - ybar - b * (x - xbar)
            sse += w * d * d
        resid_sd = math.sqrt(sse / max(sw - 2.0, 1.0))
    return (ybar, b, xbar, sxx, n_eff, resid_sd, len(points))


def rates_for(states, trophy_index, t0):
    """[(x, rate_pct, logit(rate), w)] for one trophy over the distinct states.

    The logit is precomputed once here: the ceiling grid search scores every
    candidate against it, and recomputing the log per candidate dominates
    the runtime on histories with many states.
    """
    last_t = hours_since(states[-1][0], t0) if states else 0.0
    out = []
    for when, _players, rates in states:
        if trophy_index >= len(rates):
            continue
        rate = rates[trophy_index]
        if rate is None:
            continue
        t_h = hours_since(when, t0)
        w = 0.5 ** ((last_t - t_h) / RECENCY_HALF_LIFE_H)
        out.append((log_time(when, t0), rate, logit_pct(rate), w))
    return out


def _sigmoid(z):
    if z >= 700:
        return 1.0
    if z <= -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def logistic_fit(pts, r_max, tier):
    """Saturating-curve fit for one trophy.

    Model: rate(x) = pi * sigmoid(a + b*x), x = ln(1 + hours since release),
    i.e. logit(rate/pi) is linear in x. pi (the eventual ceiling) is chosen
    by grid search: the smallest ceiling whose fit, scored in logit(rate)
    space, is within 10% of the best — parsimony against the flat likelihood
    of a ceiling far above the data. Returns (chosen, near_best), chosen =
    (pi, wls_fit) and near_best = [(sse, pi, wls_fit)] within 25% of best.
    """
    cap = TIER_PI_CAP.get(tier, 99.0)
    # A 0% reading is censored (fewer than half the 2-decimal quantum earned
    # it): it has no logit, so it cannot enter the u-fit or its score.
    fpts = [(x, r, lr, w) for (x, r, lr, w) in pts if r > 0.0]
    if len(fpts) < 2:
        return None, []
    cands = []
    for mult in PI_MULTS:
        pi = min(r_max * mult, cap)
        if pi <= r_max * 1.01:
            continue
        upts = []
        ok = True
        for (x, r, _lr, w) in fpts:
            u = r / pi
            if u <= 0.0 or u >= 1.0:
                ok = False
                break
            upts.append((x, math.log(u / (1.0 - u)), w))
        if not ok or len(upts) < 2:
            continue
        fit = wls(upts)
        if fit is None:
            continue
        # score in logit(rate) space, so ceilings compare on one scale
        ybar, b, xbar, _sxx, _ne, _rs, _n = fit
        sse = 0.0
        for (x, _r, lr, w) in fpts:
            r_hat = pi * _sigmoid(ybar + b * (x - xbar))
            d = logit_pct(r_hat) - lr
            sse += w * d * d
        cands.append((sse, pi, fit))
    if not cands:
        return None, []
    best_sse = min(s for s, _, _ in cands)
    near = [c for c in cands if c[0] <= 1.25 * best_sse]
    chosen = min((c for c in cands if c[0] <= 1.10 * best_sse),
                 key=lambda c: c[1])
    return chosen, near


def fit_all(states, trophies, t0, x_target, horizon_hours):
    """Fit every trophy; returns a list of per-trophy dicts.

    kind: 'logistic' (saturating-curve fit), 'flat' (no trend estimable:
    dilution prior), 'pinned' (rate ~100%), 'floor' (rate ~0%).
    Each dict carries m (median logit), sigma_pi (ceiling uncertainty),
    fit (the u-space WLS tuple, for leverage/spread), cnt.
    """
    n = len(trophies)
    # Per-trophy series once, reused by the fits and the flat checks below.
    pts_list = [rates_for(states, i, t0) for i in range(n)]
    # Per-trophy logistic fits first, so tier slopes can pool them.
    chosen_list = []
    near_list = []
    for i in range(n):
        pts = pts_list[i]
        r_max = max((r for (_x, r, _lr, _w) in pts), default=1.0)
        chosen, near = logistic_fit(pts, r_max, trophies[i].get("tier"))
        chosen_list.append(chosen)
        near_list.append(near)
    tier_slopes = {}
    for tier in sorted({t.get("tier") for t in trophies}):
        slopes = [c[2][1] for i, c in enumerate(chosen_list)
                  if c is not None and trophies[i].get("tier") == tier and c[2][6] >= 3]
        tier_slopes[tier] = median_of(slopes) if slopes else None
    global_slopes = [c[2][1] for c in chosen_list if c is not None and c[2][6] >= 3]
    global_slope = median_of(global_slopes) if global_slopes else 0.0
    for tier, val in tier_slopes.items():
        if val is None:
            tier_slopes[tier] = global_slope

    # Refresh rate over the states (for the flat-history dilution prior).
    if len(states) >= 2:
        span_h = max(hours_since(states[-1][0], t0) - hours_since(states[0][0], t0), 0.0)
        refresh_rate = (len(states) + REFRESH_PRIOR_N - 1.0) / (span_h + REFRESH_PRIOR_H)
    else:
        refresh_rate = REFRESH_PRIOR_N / REFRESH_PRIOR_H
    expected_refreshes = min(refresh_rate * max(horizon_hours, 0.0), 6.0)

    out = []
    for i, trophy in enumerate(trophies):
        last_rate = None
        if states:
            rates = states[-1][2]
            if i < len(rates):
                last_rate = rates[i]
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= PINNED_RATE:
            out.append({"m": logit_pct(HI), "sigma_pi": 0.0, "kind": "pinned",
                        "fit": None, "cnt": 0})
            continue
        if last_rate <= FLOOR_RATE:
            out.append({"m": logit_pct(FLOOR_RATE), "sigma_pi": 0.0, "kind": "floor",
                        "fit": None, "cnt": 0})
            continue
        pts = pts_list[i]
        flat = len(pts) < 2 or (
            max(r for (_x, r, _lr, _w) in pts) - min(r for (_x, r, _lr, _w) in pts) < 1e-9)
        if flat or chosen_list[i] is None:
            drift = REFRESH_DRIFT * expected_refreshes
            out.append({"m": logit_pct(last_rate) - drift, "sigma_pi": 0.0,
                        "kind": "flat", "fit": None, "cnt": len(pts)})
            continue
        _sse, pi, fit = chosen_list[i]
        ybar, b_raw, xbar, sxx, n_eff, resid_sd, cnt = fit
        b_tier = tier_slopes.get(trophy.get("tier"), global_slope)
        b = (n_eff * b_raw + SHRINK_PRIOR_W * b_tier) / (n_eff + SHRINK_PRIOR_W)
        b = max(-SLOPE_CAP, min(SLOPE_CAP, b))
        r_med = pi * _sigmoid(ybar + b * (x_target - xbar))
        m = logit_pct(r_med)
        # ceiling uncertainty: spread of the target rate over near-best pis
        targets = []
        for (_sse, pi2, fit2) in near_list[i]:
            y2, b2, xb2, _sx2, _ne2, _rs2, _c2 = fit2
            b2s = (n_eff * b2 + SHRINK_PRIOR_W * b_tier) / (n_eff + SHRINK_PRIOR_W)
            b2s = max(-SLOPE_CAP, min(SLOPE_CAP, b2s))
            targets.append(logit_pct(pi2 * _sigmoid(y2 + b2s * (x_target - xb2))))
        sigma_pi = (max(targets) - min(targets)) / 4.0 if len(targets) >= 2 else 0.0
        out.append({"m": m, "sigma_pi": sigma_pi, "kind": "logistic",
                    "fit": fit, "cnt": cnt})
    return out


def rolling_sigma(states, trophies, t0):
    """Pooled one-step-ahead logit error scale from rolling-origin backtests.

    Fit on states 0..k, forecast state k+1, collect the logit errors.
    Returns the robust (MAD-based) scale, or the default when too few.
    Only the most recent 8 origins are used: the loop is quadratic in the
    number of states, and the final case may hold dozens of them, so the
    cap keeps the run far inside its 8-second budget.
    """
    errors = []
    for k in range(max(2, len(states) - 9), len(states) - 1):
        prefix = states[:k + 1]
        x_next = log_time(states[k + 1][0], t0)
        horizon_h = (states[k + 1][0] - states[k][0]).total_seconds() / 3600.0
        preds = fit_all(prefix, trophies, t0, x_next, horizon_h)
        next_rates = states[k + 1][2]
        for i in range(len(trophies)):
            if i >= len(next_rates) or next_rates[i] is None:
                continue
            if preds[i]["kind"] == "pinned":
                continue
            errors.append(logit_pct(next_rates[i]) - preds[i]["m"])
    if len(errors) >= 20:
        med = median_of(errors)
        mad = median_of([abs(e - med) for e in errors])
        scale = 1.4826 * mad
        return max(scale, 0.05)
    if len(errors) >= 5:
        mean = sum(errors) / len(errors)
        return max(math.sqrt(sum((e - mean) ** 2 for e in errors) / len(errors)), 0.05)
    return SIGMA_POOL_DEFAULT


def forecast_case(case, name):
    trophies = case["trophies"]
    levels = list(case.get("levels") or LEVELS_DEFAULT)
    zs = [norm_ppf(lv) for lv in levels]
    states = distinct_states(case)
    t0 = release_time(case)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = log_time(target, t0)
    horizon_h = max((target - at).total_seconds() / 3600.0,
                    float(case.get("horizon_minutes") or 0) / 60.0)
    if not states or not trophies:
        return {"trophies": [{"id": t["id"], "q": [round4(1.0)] * len(levels)}
                             for t in trophies]}
    preds = fit_all(states, trophies, t0, x_target, horizon_h)
    sigma_pool = rolling_sigma(states, trophies, t0)
    rows = []
    for i, trophy in enumerate(trophies):
        info = preds[i]
        m = info["m"]
        kind = info["kind"]
        if kind == "logistic":
            fit = info["fit"]
            _ybar, _b, xbar, sxx, n_eff, resid_sd, cnt = fit
            lev = 1.0 + (x_target - xbar) ** 2 / sxx + 1.0 / max(n_eff, 1e-9)
            mult = 1.0
            if resid_sd is not None and sigma_pool > 0:
                var = (cnt * resid_sd * resid_sd + 5.0 * sigma_pool * sigma_pool) / (cnt + 5.0)
                mult = max(0.7, min(1.6, math.sqrt(var) / sigma_pool))
            base = sigma_pool * math.sqrt(min(lev, LEV_CAP)) * mult
            sigma = math.sqrt(base * base + info["sigma_pi"] ** 2)
            sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        elif kind == "flat":
            sigma = SIGMA_FLAT
        elif kind == "pinned":
            sigma = 0.05
        else:  # floor
            sigma = 0.80
        q = [round4(min(HI, max(LO, inv_logit_pct(m + z * sigma)))) for z in zs]
        # rounding can only break monotonicity by a hair; enforce it.
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def persistence_rows(case):
    """Starter-style fallback: last rate, random-walk spread. Never crashes."""
    trophies = case["trophies"]
    levels = list(case.get("levels") or LEVELS_DEFAULT)
    zs = [norm_ppf(lv) for lv in levels]
    states = distinct_states(case)
    spread = 0.5 * math.sqrt(max(float(case.get("horizon_minutes") or 0) / 1440.0, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        rate = 1.0
        if states and i < len(states[-1][2]) and states[-1][2][i] is not None:
            rate = states[-1][2][i]
        lg = logit_pct(rate)
        q = [round4(min(HI, max(LO, inv_logit_pct(lg + z * spread)))) for z in zs]
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
