# Rebuild + Load DualPI2 and Setup Namespaces

This guide explains what `l4s_scripts/rebuild_dualpi2.sh` does, how to configure it, and how to verify everything is working. It assumes you cloned this repo as-is (so `re_init_qdisc.sh` is in the repo root and this script is in `l4s_scripts/`).

## What this script does

1. Removes any leftover namespaces from prior runs. This script creates and uses the namespaces `ns_s` and `ns_r`; do not manually create namespaces with these names to avoid collisions during setup.
2. Unloads any currently loaded `sch_dualpi2` kernel module.
3. Builds `sch_dualpi2.ko` from the custom Linux source tree that includes the DualPI2 modifications, such as mark and drop counters.
4. Loads the freshly built module with `insmod` and ensures it is the active one. If a different copy is active, it copies your built module into the system modules directory and runs `depmod`.
5. Creates two network namespaces (`ns_s`, `ns_r`) and a `veth` pair (`veth-s` ↔ `veth-r`).
6. Assigns IPs, brings links up, and sets TCP to `prague` with ECN enabled in both namespaces.
7. Calls `../re_init_qdisc.sh` to attach an HTB root qdisc and the DualPI2 leaf on `ns_s` egress (`veth-s`).

## Why a custom Linux tree is required

This setup relies on a custom `DualPI2` qdisc that exposes additional counters/telemetry (e.g., probability, marks, delays). These changes do not alter the algorithm’s logic; they only add visibility. Because stock kernels typically lack these additions, you must build the module from a custom Linux source tree that includes the DualPI2 modifications and matches your running kernel. You can use our tree (https://github.com/gargAneesh/linux) or your own; set its path in `LINUX_PATH` inside `rebuild_dualpi2.sh`.

## Prerequisites

- A Linux source tree that includes your DualPI2 changes (see configuration below).
- Build tools and headers appropriate for building the module (the tree at `LINUX_PATH` should be configured for your running kernel).
- `sudo` privileges (the script uses namespaces, `insmod`, qdisc changes, etc.).
- A `tc` binary that recognizes the `dualpi2` qdisc:
  - In this repository, `re_init_qdisc.sh` references a custom `tc` built from https://github.com/gargAneesh/iproute2. Clone/build it and point `re_init_qdisc.sh` to the resulting `tc` binary on your device.
  - This `tc` exposes the DualPI2 counters added to the kernel. Verify with:
    `sudo ip netns exec ns_s <tc_path> -s qdisc show dev veth-s` and look for `c_marks`, `l_marks`, `c_drops`, `l_drops`.
  - If your `tc` is installed elsewhere (or your system `tc` already supports DualPI2), simply update the path in `re_init_qdisc.sh` to match your environment.

## Build tc (from our iproute2)

If you clone our `iproute2` repo to get a DualPI2-aware `tc`, you need to build it once:

```bash
git clone https://github.com/gargAneesh/iproute2
cd iproute2
make -j"$(nproc)"
# The tc binary will be at iproute2/tc/tc
```

- Then set that path in `re_init_qdisc.sh` (e.g., `/path/to/iproute2/tc/tc`).
- You do not need to build the entire Linux tree here; the `rebuild_dualpi2.sh` script builds just the `sch_dualpi2.ko` module from your custom Linux source.

<!-- Dependencies (install before `make`):

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libelf-dev bison flex libmnl-dev

# Fedora
sudo dnf install -y gcc make pkgconf-pkg-config elfutils-libelf-devel bison flex libmnl-devel

# Arch
sudo pacman -S --needed base-devel pkgconf libelf bison flex libmnl
``` -->

## Configure

Open `l4s_scripts/rebuild_dualpi2.sh` and set the path to your Linux source:

```bash
# Line near the top of the script
LINUX_PATH=<path/to/linux>
```

- Set `LINUX_PATH` to the root of the Linux source tree that contains your DualPI2 qdisc changes.
- Ensure `../re_init_qdisc.sh` exists relative to `l4s_scripts/` and, if needed, update the `tc` path inside it.

## Run

From anywhere (the script resolves paths internally):

```bash
cd l4s_scripts
./rebuild_dualpi2.sh
```

The script will:
- Delete namespaces if they already exist
- Unload any loaded `sch_dualpi2`
- Build and load your updated `sch_dualpi2.ko`
- Set up `ns_s`, `ns_r`, the `veth` pair, IPs, and sysctls
- Attach HTB + DualPI2 on `veth-s` in `ns_s`

## Verify

- Namespaces created:

```bash
ip netns ls
```

- Module is loaded and from the expected source:

```bash
lsmod | grep dualpi2
modinfo sch_dualpi2 | grep filename
```

- Qdisc attached and reporting DualPI2 stats:

```bash
sudo ip netns exec ns_s <path/to/iproute2>/tc/tc -s qdisc show dev veth-s
# If you changed tc path in re_init_qdisc.sh, use that path here too.
```

- TCP Prague & ECN settings in both namespaces:

```bash
sudo ip netns exec ns_s sysctl net.ipv4.tcp_congestion_control
sudo ip netns exec ns_s sysctl net.ipv4.tcp_ecn
sudo ip netns exec ns_r sysctl net.ipv4.tcp_congestion_control
sudo ip netns exec ns_r sysctl net.ipv4.tcp_ecn
```

## Troubleshooting

- "Module still loaded" error: remove any qdiscs using DualPI2, then re-run.
- Build errors: ensure your `LINUX_PATH` is correct and the tree is configured to build modules for your running kernel.
- `tc` errors like "Unknown qdisc": update `re_init_qdisc.sh` to point to a `tc` that includes DualPI2 support, or install one.
- Permissions: the script requires `sudo`; if any step fails with EPERM, re-run with a user that can `sudo`.

## Cleanup

To tear down the topology and unload the module:

```bash
sudo ip netns del ns_s || true
sudo ip netns del ns_r || true
sudo rmmod sch_dualpi2 || true
```

---

Notes:
- This file is specific to `rebuild_dualpi2.sh`. The repository’s original `README.md` remains unchanged for broader context. If you prefer, you can link to this doc from the main README.
