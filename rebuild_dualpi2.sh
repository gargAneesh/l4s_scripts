#!/bin/bash
set -e

# --- Configurable paths ---
MODULE_NAME=sch_dualpi2
SRC_KO_PATH=/your-path/linux/net/sched/sch_dualpi2.ko
SYS_KO_PATH=/lib/modules/$(uname -r)/kernel/net/sched/sch_dualpi2.ko

echo "[INFO] Deleting existing namespaces..."

if $(sudo ip netns list | grep -q ns_s); then
    echo "  -> Removing namespace ns_s"
    sudo ip netns del ns_s
fi
if $(sudo ip netns list | grep -q ns_r); then
    echo "  -> Removing namespace ns_r"
    sudo ip netns del ns_r
fi

# for ns in $(ip netns list | awk '{print $1}'); do
#     if sudo ip netns exec "$ns" ~/moment_lab/iproute2/tc/tc qdisc show | grep -q "dualpi2"; then
#         echo "  -> Removing namespace $ns"
#         devs=$(sudo ip netns exec "$ns" ~/moment_lab/iproute2/tc/tc qdisc show | grep "dualpi2" | awk '{print $5}')
#         for dev in $devs; do
#             sudo ip netns delete "$ns" # Remove entire namespace
#         done
#     fi
# done

echo "[INFO] Removing $MODULE_NAME if loaded..."
sudo modprobe -r $MODULE_NAME 2>/dev/null || true
sudo rmmod $MODULE_NAME 2>/dev/null || true


# Check that it's really gone
if lsmod | grep -q $MODULE_NAME; then
    echo "[ERROR] Module still loaded. Try removing dependent qdiscs manually."
    exit 1
fi

# make the module
echo "[INFO] Building the module from source..."
sudo make -C ~/moment_lab/linux M=~/moment_lab/linux/net/sched/ sch_dualpi2.ko -j$(nproc)

echo "[INFO] Loading updated module..."
sudo insmod $SRC_KO_PATH

# Verify which .ko file is actually active
loaded_path=$(modinfo $MODULE_NAME | grep filename | awk '{print $2}')
if [ "$loaded_path" == "$SRC_KO_PATH" ]; then
    echo "[SUCCESS] Loaded the updated module from source path."
else
    echo "[WARN] Loaded version differs:"
    echo "  Expected: $SRC_KO_PATH"
    echo "  Actual:   $loaded_path"
    echo "[INFO] Replacing system version to ensure consistency..."
    sudo cp "$SRC_KO_PATH" "$SYS_KO_PATH"
    sudo depmod -a
fi

# make the namespaces and assign dualpi2 to them
echo "[INFO] Creating network namespaces and assigning $MODULE_NAME qdisc..."

sudo ip netns add ns_s
sudo ip netns add ns_r

sudo ip link add veth-s type veth peer name veth-r

sudo ip link set veth-s netns ns_s
sudo ip link set veth-r netns ns_r

sudo ip netns exec ns_s ip addr add 172.20.1.1/30 dev veth-s
sudo ip netns exec ns_r ip addr add 172.20.1.2/30 dev veth-r

sudo ip netns exec ns_s ip link set lo up
sudo ip netns exec ns_r ip link set lo up

sudo ip netns exec ns_s ip link set veth-s up
sudo ip netns exec ns_r ip link set veth-r up

sudo ip netns exec ns_s sysctl -w net.ipv4.tcp_congestion_control=prague 
sudo ip netns exec ns_r sysctl -w net.ipv4.tcp_congestion_control=prague

sudo ip netns exec ns_r sysctl -w net.ipv4.tcp_ecn=3
sudo ip netns exec ns_s sysctl -w net.ipv4.tcp_ecn=3

echo "[INFO] Attach HTB and DualPI2 on sender egress interface (veth-s)"
~/moment_lab/Scripts/re_init_qdisc.sh

echo "[INFO] Checking kernel messages for printk output in ns_s..."
# sudo ip netns exec ns_s dmesg -T | grep dualpi2

