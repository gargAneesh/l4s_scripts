echo "[INFO] Re-initializing qdisc setup in namespaces..."

# 1) Root HTB qdisc
sudo ip netns exec ns_s /home/aneesh/moment_lab/iproute2/tc/tc qdisc del dev veth-s root 2>/dev/null

sudo ip netns exec ns_s /home/aneesh/moment_lab/iproute2/tc/tc qdisc add dev veth-s root handle 1: htb default 10

# 2) Add the class under HTB
sudo ip netns exec ns_s /home/aneesh/moment_lab/iproute2/tc/tc class add dev veth-s parent 1: classid 1:10 htb rate 12mbit ceil 12mbit burst 15k

# 3) Attach DualPI2 as the leaf qdisc
sudo ip netns exec ns_s /home/aneesh/moment_lab/iproute2/tc/tc qdisc add dev veth-s parent 1:10 handle 2: dualpi2 target 5ms step_thresh 1ms limit 200
