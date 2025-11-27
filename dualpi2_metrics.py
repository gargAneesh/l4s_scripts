#!/usr/bin/env python3
"""
DualPI2 Metrics Collection Tool
================================

This script automates the collection and plotting of DualPI2 qdisc and TCP metrics from network namespaces using iperf3, ss, and tc utilities.

Features:
---------
- Runs L4S and/or classic flows between sender/receiver namespaces using iperf3 (TCP Prague and UDP).
- Periodically samples metrics using `ss -tin` and `tc -s qdisc show`.
- Records high-resolution timestamps for each sample.
- Writes all metrics and timestamps to a CSV file.
- Generates time-series plots for delays/rtt, credit, probability, and ECN/step marks.

Usage:
------
1. Ensure you have the required namespaces (`ns_s`, `ns_r`), veth device, and iproute2/tc path configured.
2. Run the script with Python 3:
    python3 dualpi2_metrics.py [--flows both|l4s|classic]
    - Default is both flows (L4S and classic).
    - Use --flows l4s to run only L4S (TCP Prague).
    - Use --flows classic to run only classic (UDP by default).
    - Add -T / --classic-tcp to send classic over TCP instead of UDP.
3. After completion, CSV and PNG plots are saved in the current directory.

Example:
    # Run both flows; classic over UDP (default)
    python3 dualpi2_metrics.py --flows both

    # Run both flows; classic over TCP
    python3 dualpi2_metrics.py --flows both -T

    # Classic-only over TCP
    python3 dualpi2_metrics.py --flows classic -T

What it does:
-------------
- Starts iperf3 servers in sender namespace(s) on ports 5201 (L4S) and/or 5202 (classic).
- Starts iperf3 clients in receiver namespace(s) for L4S (TCP Prague) and/or classic (UDP/TCP).
- Every INTERVAL seconds, collects TCP and qdisc metrics, timestamps, and writes to CSV.
- After run, generates:
     - delays/rtt plot (_delays_rtt.png)
     - credit plot (_credit.png)
     - probability plot (_probability.png)
     - ECN/step marks plot (_marks.png)

Dependencies:
-------------
- Python 3
- iperf3, iproute2/tc, ss utilities
- pandas, matplotlib (for plotting)

See script configuration section for tunable parameters.
"""

import subprocess
import time
import re
import csv
import signal
from datetime import datetime
import os
import argparse

# optional plotting libs
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    plt = None

# === CONFIGURATION ===

INTERVAL = 0.5                 # seconds between samples
VETH_DEVICE = "veth-s"         # egress interface with DualPI2
DST_IP = "172.20.1.2"          # receiver IP
NS_SENDER = "ns_s"
NS_RECEIVER = "ns_r"
TC_PATH = "/home/aneesh/moment_lab/iproute2/tc/tc"       # adjust if needed
DURATION = 60                  # iperf3 run time in seconds
CC_ALGO = "prague"             # congestion control algorithm
PORT_L4S = 5201
PORT_CLASSIC = 5202
# OUTFILE will be constructed dynamically based on --outdir
# You can override the default output directory via:
#   1) CLI: --outdir /desired/path
#   2) Env var: export DUALPI2_OUTDIR=/desired/path (used if --outdir not given)
OUTDIR_DEFAULT = os.environ.get("DUALPI2_OUTDIR", "/home/aneesh/moment_lab/test_data")


# === UTILITY FUNCTIONS ===
def run(cmd, **kwargs):
    """Run a command and return output as string."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs).stdout



def start_iperf_servers(flows):
    """Start iperf3 servers for selected flows (in ns_r)."""
    procs = []
    if flows in ('both', 'l4s'):
        print("[+] Starting L4S iperf3 server in ns_r (port 5201)...")
        procs.append(subprocess.Popen([
            "sudo", "ip", "netns", "exec", NS_RECEIVER, "iperf3", "-s", "-p", str(PORT_L4S)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    if flows in ('both', 'classic'):
        print("[+] Starting classic iperf3 server in ns_r (port 5202)...")
        procs.append(subprocess.Popen([
            "sudo", "ip", "netns", "exec", NS_RECEIVER, "iperf3", "-s", "-p", str(PORT_CLASSIC)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs

def start_iperf_clients(flows,
                        classic_tcp: bool = False,
                        start_delay: float = 0.0,
                        flow_order: str = 'l4s-first',
                        base_duration: int = DURATION):
    """Start iperf3 client flows in the sender namespace.

    Behavior:
      - Single flow (l4s or classic): run exactly base_duration seconds.
      - Both flows: start first flow immediately, wait start_delay seconds, start second flow.
        The first flow's iperf duration is extended by start_delay so both flows overlap for
        a full base_duration after the second flow starts.

    Returns:
        (process_list, l4s_start_ts_ns, classic_start_ts_ns)
    """
    procs = []
    l4s_start_ts_ns = None
    classic_start_ts_ns = None

    def _start_l4s(run_duration):
        print(f"[+] Starting L4S iperf3 client in ns_s (port {PORT_L4S}, CC={CC_ALGO}, t={run_duration})...")
        return subprocess.Popen([
            "sudo", "ip", "netns", "exec", NS_SENDER,
            "iperf3", "-c", DST_IP, "-p", str(PORT_L4S), "-t", str(run_duration), "-C", CC_ALGO
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _start_classic(run_duration):
        if classic_tcp:
            print(f"[+] Starting classic iperf3 client in ns_s (port {PORT_CLASSIC}, TCP, t={run_duration})...")
            return subprocess.Popen([
                "sudo", "ip", "netns", "exec", NS_SENDER,
                "iperf3", "-c", DST_IP, "-p", str(PORT_CLASSIC), "-t", str(run_duration)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        else:
            print(f"[+] Starting classic iperf3 client in ns_s (port {PORT_CLASSIC}, UDP, t={run_duration})...")
            return subprocess.Popen([
                "sudo", "ip", "netns", "exec", NS_SENDER,
                "iperf3", "-c", DST_IP, "-p", str(PORT_CLASSIC), "-u", "-t", str(run_duration)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if flows == 'l4s':
        p = _start_l4s(base_duration)
        procs.append(p)
        l4s_start_ts_ns = time.time_ns()
    elif flows == 'classic':
        p = _start_classic(base_duration)
        procs.append(p)
        classic_start_ts_ns = time.time_ns()
    else:  # both flows
        # Extend duration of first so total overlap after second start is base_duration.
        overlap_extension = start_delay if start_delay > 0 else 0.0
        first_duration = base_duration + int(overlap_extension)  # keep iperf integer seconds
        second_duration = base_duration
        if flow_order == 'l4s-first':
            p1 = _start_l4s(first_duration)
            l4s_start_ts_ns = time.time_ns()
            procs.append(p1)
            if start_delay > 0.0:
                print(f"[i] Waiting {start_delay:.3f}s before starting classic flow...")
                time.sleep(start_delay)
            p2 = _start_classic(second_duration)
            classic_start_ts_ns = time.time_ns()
            procs.append(p2)
        else:
            p1 = _start_classic(first_duration)
            classic_start_ts_ns = time.time_ns()
            procs.append(p1)
            if start_delay > 0.0:
                print(f"[i] Waiting {start_delay:.3f}s before starting L4S flow...")
                time.sleep(start_delay)
            p2 = _start_l4s(second_duration)
            l4s_start_ts_ns = time.time_ns()
            procs.append(p2)

    print(f"[i] Flow order: {flow_order} | Start delay: {start_delay:.3f}s")
    return procs, l4s_start_ts_ns, classic_start_ts_ns


def parse_ss(output):
    """Extract TCP metrics from `ss -tin` output."""
    metrics = {}
    patterns = {
        "rtt": r"rtt:(\d+\.\d+)",
        "pacing_rate": r"pacing_rate\s+(\d+)",
        "delivery_rate": r"delivery_rate\s+(\d+)",
        "cwnd": r"cwnd:(\d+)",
        "bytes_acked": r"bytes_acked:(\d+)",
        "bytes_sent": r"bytes_sent:(\d+)",
    }
    for k, p in patterns.items():
        m = re.search(p, output)
        metrics[k] = float(m.group(1)) if m else None
    return metrics


def parse_tc(output):
    """Extract DualPI2 metrics from `tc -s qdisc show` output."""
    metrics = {}
    # alpha, beta, prob are floating fields
    for f in ["alpha", "beta", "prob"]:
        m = re.search(rf"\b{f}\s+([\d\.eE+-]+)", output)
        metrics[f] = float(m.group(1)) if m else None

    # delays are reported like: delay_c 5277us or delay_l 0us
    def _parse_delay(field_name):
        # match digits possibly followed by unit letters (e.g., '5277us' or '5ms')
        m = re.search(rf"\b{field_name}\s+([\d\.]+)([a-zA-Z]+)?", output)
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2) or "us"
        unit = unit.lower()
        # normalize to microseconds
        if unit == 'us':
            return val
        if unit == 'ms':
            return val * 1000.0
        if unit == 's':
            return val * 1e6
        if unit == 'ns':
            return val / 1000.0
        # fallback: return raw value
        return val

    metrics['delay_C'] = _parse_delay('delay_c')
    metrics['delay_L'] = _parse_delay('delay_l')

    # credit may be negative and may include trailing text like '(L)'
    m = re.search(r"\bcredit\s+(-?\d+)", output)
    metrics['credit'] = int(m.group(1)) if m else None

    # other integer fields (base counters)
    for f in ['ecn_mark', 'pkts_in_c', 'pkts_in_l', 'step_marks', 'dq_count', 'eq_count']:
        m = re.search(rf"\b{f}\s+(\d+)", output)
        metrics[f] = int(m.group(1)) if m else None
    # optional per-queue PI2 ECN mark counters appended in extended xstats
    for f in ['c_marks', 'l_marks']:
        m = re.search(rf"\b{f}\s+(\d+)", output)
        metrics[f] = int(m.group(1)) if m else None

    return metrics


def capture_metrics():
    """Run ss and tc commands, parse metrics."""
    # Build command lists to avoid extra shell overhead
    ss_cmd_list = ["sudo", "ip", "netns", "exec", NS_SENDER, "ss", "-tin", "dst", DST_IP]
    tc_cmd_list = ["sudo", "ip", "netns", "exec", NS_SENDER, TC_PATH, "-s", "qdisc", "show", "dev", VETH_DEVICE]

    t0 = time.time_ns()
    ss_proc = subprocess.run(ss_cmd_list, capture_output=True, text=True)
    ss_out = ss_proc.stdout
    tc_proc = subprocess.run(tc_cmd_list, capture_output=True, text=True)
    tc_out = tc_proc.stdout
    t1 = time.time_ns()

    # Use midpoint as the best estimate of when the snapshot represents
    ts_ns = (t0 + t1) // 2

    data = parse_ss(ss_out)
    data.update(parse_tc(tc_out))
    # record both ISO timestamp (ms) and high-resolution epoch ns
    data["timestamp_ns"] = int(ts_ns)
    data["timestamp"] = datetime.fromtimestamp(ts_ns / 1e9).isoformat(timespec="milliseconds")
    # duration of the capture commands (ms)
    data["cmd_duration_ms"] = (t1 - t0) / 1e6
    return data



def main():
    parser = argparse.ArgumentParser(description="DualPI2 metrics collection")
    parser.add_argument('--flows', choices=['both', 'l4s', 'classic'], default='both', help='Which flows to run (default: both)')
    parser.add_argument('-T', '--classic-tcp', action='store_true', help='Use TCP for the classic flow (default: UDP)')
    parser.add_argument('--outdir',
                        default=OUTDIR_DEFAULT,
                        help='Directory for CSV and plot outputs (default: %(default)s; env override: DUALPI2_OUTDIR)')
    parser.add_argument('--start-delay', type=float, default=0.0,
                        help='Delay in seconds between starting two flows when --flows both (default: 0.0 = simultaneous)')
    parser.add_argument('--flow-order', choices=['l4s-first', 'classic-first'], default='l4s-first',
                        help='Which flow starts first when --flows both (default: l4s-first)')
    parser.add_argument('--debug-samples', type=int, default=0,
                        help='Print debug info for first N samples (0=disabled)')
    args = parser.parse_args()
    flows = args.flows
    classic_tcp = args.classic_tcp
    outdir = args.outdir
    start_delay = max(0.0, float(args.start_delay))
    flow_order = args.flow_order

    # Ensure outdir exists
    os.makedirs(outdir, exist_ok=True)
    print(f"[i] Output directory: {outdir}")

    # Build base filename prefix and full CSV path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_prefix = f"dualpi2_metrics_{timestamp}"
    outfile = os.path.join(outdir, base_prefix + '.csv')

    # Startup summary
    l4s_desc = f"TCP (CC={CC_ALGO})"
    classic_desc = "TCP" if classic_tcp else "UDP"
    print(f"[i] L4S flow: {l4s_desc} | Classic flow: {classic_desc} | Flows: {flows}")

    # Start iperf3 servers
    server_procs = start_iperf_servers(flows)
    time.sleep(1)

    # Prepare CSV (append metadata columns for flow ordering and start timestamps)
    fields = [
        "timestamp", "timestamp_ns", "scheduled_timestamp_ns", "sample_index", "rel_ns", "cmd_duration_ms",
        "rtt", "pacing_rate", "delivery_rate", "cwnd",
        "bytes_acked", "bytes_sent", "alpha", "beta", "prob",
        "delay_C", "delay_L", "credit", "ecn_mark", "step_marks", "pkts_in_c", "pkts_in_l",
        "c_marks", "l_marks",  # optional extended counters
        "dq_count", "eq_count",  # raw enqueue/dequeue counters
        "dq_delta", "eq_delta",  # per-sample differences
        "start_delay", "flow_order", "l4s_start_ts_ns", "classic_start_ts_ns"  # run metadata
    ]
    f = open(outfile, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()

    # Start iperf3 clients using refactored function
    client_procs, l4s_start_ts_ns, classic_start_ts_ns = start_iperf_clients(
        flows, classic_tcp, start_delay=start_delay, flow_order=flow_order
    )

    # Second flow start timestamp used for duration guard
    if flows == 'both':
        second_flow_start_ts_ns = classic_start_ts_ns if flow_order == 'l4s-first' else l4s_start_ts_ns
    else:
        second_flow_start_ts_ns = l4s_start_ts_ns or classic_start_ts_ns

    # Brief health check after launch and delay
    time.sleep(1.0)
    early_exit = [p for p in client_procs if p.poll() is not None]
    if early_exit:
        print(f"[!] Warning: {len(early_exit)} client(s) exited early. They may have failed to connect. Continuing metrics collection regardless.")

    print(f"[+] Monitoring metrics for flows: {flows} (Ctrl+C to stop early)")
    try:
        # Align start reference to the moment the first client was started
        start_monotonic_ns = time.monotonic_ns()
        epoch_ref_ns = time.time_ns()
        sample_index = 0
        next_sample_ns = start_monotonic_ns


        # Initialize previous counters for delta computation
        prev_dq = None
        prev_eq = None

        UINT32_MAX = 2**32
        def diff_u32(prev, curr):
            if prev is None or curr is None:
                return None
            if curr >= prev:
                return curr - prev
            # wrap-around (32-bit)
            return (curr + UINT32_MAX) - prev

        # Monitor until processes end AND at least DURATION seconds elapsed after second flow start.
        min_end_ns = (second_flow_start_ts_ns or time.time_ns()) + int(DURATION * 1e9)
        debug_limit = int(args.debug_samples)
        while any([p.poll() is None for p in client_procs]) or time.time_ns() < min_end_ns:
            now_mono = time.monotonic_ns()
            if now_mono < next_sample_ns:
                time.sleep((next_sample_ns - now_mono) / 1e9)

            row = capture_metrics()
            scheduled_ts_ns = int(epoch_ref_ns + sample_index * INTERVAL * 1e9)
            row["scheduled_timestamp_ns"] = scheduled_ts_ns
            row["sample_index"] = sample_index
            row["rel_ns"] = int(row["timestamp_ns"] - scheduled_ts_ns)

            # Compute deltas for enqueue/dequeue
            curr_dq = row.get("dq_count")
            curr_eq = row.get("eq_count")
            row["dq_delta"] = diff_u32(prev_dq, curr_dq)
            row["eq_delta"] = diff_u32(prev_eq, curr_eq)
            prev_dq, prev_eq = curr_dq, curr_eq

            # Add run-level metadata (constant values per row)
            row["start_delay"] = start_delay
            row["flow_order"] = flow_order
            row["l4s_start_ts_ns"] = l4s_start_ts_ns
            row["classic_start_ts_ns"] = classic_start_ts_ns
            if debug_limit > 0 and sample_index < debug_limit:
                print(f"[dbg] sample={sample_index} rtt={row.get('rtt')} delay_C={row.get('delay_C')} delay_L={row.get('delay_L')} credit={row.get('credit')} prob={row.get('prob')}")
            writer.writerow(row)
            f.flush()

            sample_index += 1
            next_sample_ns += int(INTERVAL * 1e9)
        # Clean up any lingering clients (they should have exited naturally)
        for p in client_procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.terminate()
        print("[+] Data collection finished (ensured full duration after second start).")
    except KeyboardInterrupt:
        print("[!] Experiment interrupted by user.")
        for p in client_procs:
            p.send_signal(signal.SIGINT)
    finally:
        # Stop iperf3 servers
        for p in server_procs:
            p.terminate()
        f.close()
        print(f"[+] Data saved to {outfile}")
        # If pandas + matplotlib available, produce plots
        if pd is None or plt is None:
            print("[i] pandas/matplotlib not available — skipping plot generation")
            return

        try:
            print("[+] Generating plots...")
            df = pd.read_csv(outfile)
            if 'timestamp_ns' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp_ns'], unit='ns')
            else:
                df['ts'] = pd.to_datetime(df['timestamp'])

            df = df.sort_values('ts')
            df.set_index('ts', inplace=True)

            # Plot delays + rtt
            delay_cols = [c for c in ['delay_C', 'delay_L', 'rtt'] if c in df.columns]
            if delay_cols:
                plt.figure(figsize=(12, 6))
                for col in delay_cols:
                    plt.plot(df.index, df[col], label=col)
                plt.xlabel('time')
                plt.ylabel('microseconds / ms')
                plt.title('Delays and RTT over time')
                plt.legend()
                plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_delays_rtt.png')
                plt.tight_layout()
                plt.savefig(png)
                plt.close()
                print(f"[+] Plot written to {png}")
            # Plot credit
            if 'credit' in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df.index, df['credit'], label='credit', color='tab:blue')
                plt.xlabel('time')
                plt.ylabel('credit')
                plt.title('Credit over time')
                plt.legend()
                plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_credit.png')
                plt.tight_layout()
                plt.savefig(png)
                plt.close()
                print(f"[+] Plot written to {png}")
            # Plot probability only
            if 'prob' in df.columns:
                plt.figure(figsize=(12, 4))
                plt.plot(df.index, df['prob'], label='probability', color='tab:red')
                plt.xlabel('time')
                plt.ylabel('probability')
                plt.title('Probability over time')
                plt.legend()
                plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_probability.png')
                plt.tight_layout()
                plt.savefig(png)
                plt.close()
                print(f"[+] Plot written to {png}")
            # Plot mark counters together (any present among ecn_mark, step_marks, c_marks, l_marks)
            mark_cols = [c for c in ['ecn_mark', 'step_marks', 'c_marks', 'l_marks'] if c in df.columns]
            if mark_cols:
                plt.figure(figsize=(12, 4))
                color_map = {
                    'ecn_mark': 'tab:green',
                    'step_marks': 'tab:orange',
                    'c_marks': 'tab:blue',
                    'l_marks': 'tab:purple'
                }
                for col in mark_cols:
                    plt.plot(df.index, df[col], label=col, color=color_map.get(col))
                plt.xlabel('time')
                plt.ylabel('count')
                plt.title('Mark Counters over time')
                plt.legend()
                plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_marks.png')
                plt.tight_layout()
                plt.savefig(png)
                plt.close()
                print(f"[+] Plot written to {png}")
        except Exception as e:
            print(f"[!] Plot generation failed: {e}")
            return

        # Additional plots for enqueue/dequeue raw and deltas
        try:
            # Raw counters
            if 'dq_count' in df.columns or 'eq_count' in df.columns:
                plt.figure(figsize=(12,4))
                if 'dq_count' in df.columns:
                    plt.plot(df.index, df['dq_count'], label='dq_count (raw)', color='tab:brown')
                if 'eq_count' in df.columns:
                    plt.plot(df.index, df['eq_count'], label='eq_count (raw)', color='tab:olive')
                plt.xlabel('time')
                plt.ylabel('count')
                plt.title('Enqueue / Dequeue Raw Counters')
                plt.legend(); plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_enqueue_dequeue_raw.png')
                plt.tight_layout(); plt.savefig(png); plt.close()
                print(f"[+] Plot written to {png}")

            # Delta counters
            if 'dq_delta' in df.columns or 'eq_delta' in df.columns:
                plt.figure(figsize=(12,4))
                if 'dq_delta' in df.columns:
                    plt.plot(df.index, df['dq_delta'], label='dq_delta', color='tab:brown')
                if 'eq_delta' in df.columns:
                    plt.plot(df.index, df['eq_delta'], label='eq_delta', color='tab:olive')
                plt.xlabel('time')
                plt.ylabel('packets per sample')
                plt.title('Enqueue / Dequeue Per-Sample Deltas')
                plt.legend(); plt.grid(True)
                png = os.path.join(outdir, base_prefix + '_enqueue_dequeue_delta.png')
                plt.tight_layout(); plt.savefig(png); plt.close()
                print(f"[+] Plot written to {png}")
        except Exception as e:
            print(f"[!] Additional enqueue/dequeue plot generation failed: {e}")




if __name__ == "__main__":
    main()

