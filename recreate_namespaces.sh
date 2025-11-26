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
