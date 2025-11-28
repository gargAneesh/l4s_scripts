# DualPI2 Metrics Collection Tool

Automates collection and plotting of DualPI2 qdisc and TCP metrics from Linux network namespaces using `iperf3`, `ss`, and `tc`.

## Features
- Run L4S (TCP Prague) and/or classic (UDP or TCP) flows.
- Staggered flow start with configurable delay and order.
- Periodic sampling (`ss -tin`, `tc -s qdisc show`) with high-resolution timestamps.
- CSV logging of qdisc + TCP metrics, including per-sample enqueue/dequeue deltas.
- Automatic generation of plots (if `pandas` + `matplotlib` installed): delays/RTT, credit, probability, mark counters, enqueue/dequeue raw & delta.

## Requirements
- Python 3
- `iperf3`, `iproute2/tc`, `ss` utilities available and accessible in namespaces
- Namespaces: sender `ns_s`, receiver `ns_r`, and veth device with DualPI2 (`veth-s` by default)
- Optional: `pandas`, `matplotlib` for plotting

## Configuration Defaults
```
INTERVAL = 0.5            # seconds between metric samples
DURATION = 60             # iperf3 client duration per flow (seconds)
VETH_DEVICE = "veth-s"
DST_IP = "172.20.1.2"
TC_PATH = "your-directory/iproute2/tc/tc"
PORT_L4S = 5201
PORT_CLASSIC = 5202
CC_ALGO = "prague"
```
Output directory defaults to env `DUALPI2_OUTDIR` or `/home/aneesh/moment_lab/test_data`.

## CLI Usage
```
python3 dualpi2_metrics.py [--flows both|l4s|classic] [-T] \
	[--start-delay SECONDS] [--flow-order l4s-first|classic-first] \
	[--debug-samples N] [--outdir PATH]
```

### Flags
- `--flows`: Which flows to run (`both` | `l4s` | `classic`). Default: `both`.
- `-T, --classic-tcp`: Send classic flow over TCP instead of UDP.
- `--start-delay`: Delay (seconds) between starting first and second client when `--flows both`. Default: `0.0` (simultaneous). Extends first flow duration so the overlap period after the second starts lasts `DURATION` seconds.
- `--flow-order`: Which flow starts first when both run (`l4s-first` | `classic-first`). Default: `l4s-first`.
- `--debug-samples`: Print parsed metric values for the first N samples (diagnostics). Default: `0` (disabled).
- `--outdir`: Where to write CSV + plots.

### Examples
Run both flows simultaneously (current default behavior):
```
python3 dualpi2_metrics.py --flows both
```
Start L4S, wait 2 seconds, then start classic (UDP):
```
python3 dualpi2_metrics.py --flows both --start-delay 2.0 --flow-order l4s-first
```
Start classic first (TCP), then L4S after 1.5s:
```
python3 dualpi2_metrics.py --flows both -T --start-delay 1.5 --flow-order classic-first
```
Classic only over TCP:
```
python3 dualpi2_metrics.py --flows classic -T
```
L4S only:
```
python3 dualpi2_metrics.py --flows l4s
```

## CSV Output
Each row contains metrics plus run metadata columns appended:
- Timing: `timestamp`, `timestamp_ns`, `scheduled_timestamp_ns`, `sample_index`, `rel_ns`, `cmd_duration_ms`
- TCP: `rtt`, `pacing_rate`, `delivery_rate`, `cwnd`, `bytes_acked`, `bytes_sent`
- DualPI2: `alpha`, `beta`, `prob`, `delay_C`, `delay_L`, `credit`, `ecn_mark`, `step_marks`, `pkts_in_c`, `pkts_in_l`, `c_marks`, `l_marks`
- Queue counters: `dq_count`, `eq_count` and per-sample deltas `dq_delta`, `eq_delta`
- Run metadata: `start_delay`, `flow_order`, `l4s_start_ts_ns`, `classic_start_ts_ns`

Notes:
- `delay_C` and `delay_L` normalized to microseconds.
- Deltas account for potential 32-bit wrap-around.
- If a flow launches later (due to `--start-delay`), early samples may reflect single-flow dynamics.

## Plots Generated
(If libraries available)
- `_delays_rtt.png`: `delay_C`, `delay_L`, `rtt`
- `_credit.png`: `credit`
- `_probability.png`: `prob`
- `_marks.png`: mark counters (`ecn_mark`, `step_marks`, `c_marks`, `l_marks`)
- `_enqueue_dequeue_raw.png`: `dq_count`, `eq_count`
- `_enqueue_dequeue_delta.png`: `dq_delta`, `eq_delta`

## Staggered Start & Duration Guarantee
When `--flows both`:
1. First client (chosen by `--flow-order`) starts immediately.
2. If `--start-delay > 0`, the script waits that many seconds.
3. Second client starts.
4. First client's iperf run time is extended by `start_delay` seconds (integer truncated) to ensure at least `DURATION` seconds of overlap after the second flow starts.
5. Sampling continues until: (all iperf clients have exited) AND (at least `DURATION` seconds have elapsed since the second flow started).

Result: The measurement window covers a full `DURATION` with both flows active (assuming they connect successfully). Total wall time ≈ `DURATION + start_delay`.

Negative delays are clamped to `0.0`.

## Tips
- Increase `DURATION` if additional post-warmup steady-state data is needed.
- Use `l4s_start_ts_ns` / `classic_start_ts_ns` to segment analysis before and after second flow arrival.
- If TCP metrics are `None`, verify the `ss` command shows a matching socket and that DualPI2 qdisc is attached to `veth-s`.
- Use `--debug-samples` for quick parsing diagnostics.
- If plots not generated, install `pandas` and `matplotlib`.

## Quick Install (optional libs)
```
pip install pandas matplotlib
```

## Future Ideas
- Separate per-flow TCP parsing if multiple TCP sockets present.
- Optional JSON summary for automation pipelines.

## License / Attribution
Internal lab usage; no explicit license header added.

---
For questions or enhancements, modify `dualpi2_metrics.py` directly.
