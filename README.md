# GTA VI Trophy Rarity Forecaster

Entry for `gta6-trophy-rarity-forecast-01` (make.fun, $250 pot). Forecasts, for
each of the 55 GTA VI PS5 trophies, five quantiles of its Exophase earn rate
at the case target. Python 3.13, standard library only, no network.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` per case:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

- **Distinct page states, not raw reads.** Exophase's page is a snapshot: it
  stays frozen between refreshes and jumps at each refresh, when newly
  tracked players dilute most rates. Consecutive identical reads collapse to
  one state each; the signal lives in the refreshes.
- **Saturating curve per trophy.** `rate(x) = pi * sigmoid(a + b*x)` with
  `x = ln(1 + hours since release)`. The ceiling `pi` is grid-searched per
  trophy with a parsimony rule (smallest ceiling within 10% of the best fit);
  the slope `b` is shrunk toward the tier's pooled slope. A straight logit
  line over the whole window overshoots badly — the growth curve is concave
  in log-time, so its chord slope is far above its forward slope.
- **Flat history gets a dilution prior**, not persistence: a refresh before
  the target is expected and pulls rates down ~0.30 logit each.
- **Honest spread.** Rolling-origin backtests inside the history pool the
  logit errors into a residual scale, widened by extrapolation leverage and
  by the spread of the target rate over near-best ceilings; floored at 0.30
  logit. A trophy pinned at ~100% stays ~100%; a trophy at ~0% gets a low
  median with a wide upward spread.
- Deterministic, ~0.1 s per practice case, ~0.4 s on a 40-state history
  (limit 8 s).

## Validation (backtest.py, entry dir — not part of the repo)

Pinball loss on the logit scale, lower is better:

| test | this entry | starter (persistence) | trend baseline |
| --- | --- | --- | --- |
| synthetic final-like case (4.5-day horizon) | **0.204** | 0.359 | 0.260 |
| practice cases vs the known refresh outcome | **0.158** | 0.198 | 0.198 |

On frozen cut cases the starter wins narrowly (0.014 vs 0.124) — nothing
moves, so the narrowest spread wins — but the real practice answers are
post-refresh, and the final case has a 4.5-day horizon with many refreshes,
where the trend model wins clearly.

## License

MIT.
