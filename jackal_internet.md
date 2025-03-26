1. Assign a Compatible IP Address to enp4s0f2

To communicate with the Jackal on the 192.168.131.x subnet, you can assign an IP address within that range to your PC's enp4s0f2 interface.

Run the following command on your PC:

sudo ip addr add 192.168.131.2/24 dev enp4s0f2

This assigns the IP address 192.168.131.2 to your PC's enp4s0f2 interface, which is compatible with the Jackal's IP address (192.168.131.1).
2. Test Connectivity

After assigning the IP, test the connection between the PC and the Jackal by pinging the Jackal from your PC:

ping 192.168.131.1

And from the Jackal, you can try pinging the PC:


ping 192.168.131.2



##### Internet sharing

1. Check IP Forwarding on the PC

Make sure IP forwarding is enabled on your PC. This allows the traffic from the Jackal (on the enp4s0f2 interface) to be forwarded to the internet through your PC's eno1 interface.

Check if IP forwarding is enabled:

bash

sudo sysctl net.ipv4.ip_forward

If it returns 0, enable IP forwarding:

sudo sysctl -w net.ipv4.ip_forward=1

Make it permanent by adding it to /etc/sysctl.conf:

net.ipv4.ip_forward=1

2. Verify NAT Configuration (Network Address Translation)

You need to ensure that your PC is correctly translating the traffic from the Jackal so it can access the internet.

Run the following commands on your PC to set up NAT:

sudo iptables -t nat -A POSTROUTING -o eno1 -j MASQUERADE
sudo iptables -A FORWARD -i enp4s0f2 -o eno1 -j ACCEPT
sudo iptables -A FORWARD -i eno1 -o enp4s0f2 -m state --state RELATED,ESTABLISHED -j ACCEPT

These rules do the following:

    MASQUERADE: This makes your PC masquerade the traffic from the Jackal as if it's coming from your PC's eno1 interface.
    Forward rules: These allow traffic to pass between enp4s0f2 (Jackal) and eno1 (internet).

3. Check Default Gateway on the Jackal

It looks like the Jackal is trying to use the wrong gateway (172.16.0.1). This could be due to incorrect routing on the Jackal.

You need to ensure that the Jackal is using the PC as its default gateway. To check the default gateway on the Jackal, run:


ip route show

You should see something like this:


default via 192.168.131.2 dev br0

If the default gateway is missing or incorrect, add it:


sudo ip route add default via 192.168.131.2 dev br0

This tells the Jackal to route all non-local traffic through your PC's enp4s0f2 interface (192.168.131.2).
4. Check Firewall Rules

If you are using a firewall (such as ufw or firewalld), ensure that it allows forwarding traffic between enp4s0f2 and eno1. You may need to allow forwarding between these interfaces.

For ufw, you can run:

bash

sudo ufw allow in on enp4s0f2
sudo ufw allow out on eno1

5. Verify Internet Connectivity

Once the configuration is complete, you should try pinging an external IP (like Google's DNS server) from the Jackal again:

bash

ping 8.8.8.8



####

1. Manually Set Reliable DNS Servers

First, we'll manually set reliable DNS servers (such as Google's 8.8.8.8 and 8.8.4.4) for your active network interface (wlp3s0).
a. Set DNS Servers Using resolvectl

    Set Google's DNS Servers:

    bash

sudo resolvectl dns wlp3s0 8.8.8.8 8.8.4.4

Alternatively, to use Cloudflare's DNS servers:

bash

sudo resolvectl dns wlp3s0 1.1.1.1 1.0.0.1

Set the DNS Domain to Unbounded:

This ensures that all DNS queries use the specified DNS servers.

bash

sudo resolvectl domain wlp3s0 ~.

Verify the Changes:

bash

resolvectl status

Expected Output:

Under Link 5 (wlp3s0), you should see:

markdown

DNS Servers: 8.8.8.8
             8.8.4.4

Or, if using Cloudflare:

markdown

DNS Servers: 1.1.1.1
             1.0.0.1

Test DNS Resolution:

bash

    ping -c 3 google.com

    Expected Output:

    You should receive replies from Google's IP addresses, indicating successful DNS resolution.

b. Make DNS Changes Persistent with Netplan

The changes made via resolvectl are temporary and will be lost upon network restart or system reboot. To make these changes persistent, configure DNS servers in Netplan.

    Identify Netplan Configuration File:

    List the Netplan configuration files:

    bash

ls /etc/netplan/

Common filenames include 01-netcfg.yaml, 50-cloud-init.yaml, etc.

Edit Netplan Configuration:

Open the relevant YAML file with a text editor (e.g., nano). Replace 01-netcfg.yaml with your actual file name if different.

bash

sudo nano /etc/netplan/01-netcfg.yaml

Add DNS Servers:

Modify the file to include the nameservers section under the relevant interface (wlp3s0). Ensure proper indentation as YAML is sensitive to it.

Example Configuration:

yaml

network:
  version: 2
  renderer: networkd
  wifis:
    wlp3s0:
      dhcp4: yes
      access-points:
        "Your_SSID":
          password: "Your_Password"
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]

    Replace "Your_SSID" and "Your_Password" with your actual Wi-Fi SSID and password.
    Ensure that wlp3s0 matches your actual wireless interface name.

Apply Netplan Configuration:

Save the file (Ctrl + O in nano), then exit (Ctrl + X).

Apply the changes:

bash

sudo netplan apply

Verify DNS Configuration:

Check if /etc/resolv.conf is correctly pointing to systemd-resolved and reflects the new DNS servers.

bash

cat /etc/resolv.conf

Expected Output:

vbnet

# This file is managed by man:systemd-resolved(8). Do not edit.
...
nameserver 127.0.0.53
options edns0 trust-ad

Use resolvectl to confirm the DNS servers:

bash

resolvectl status

Ensure that Link 5 (wlp3s0) lists 8.8.8.8 and 8.8.4.4 as DNS servers.

Test DNS Resolution Again:

bash

    ping -c 3 ubuntu.com

    You should now receive successful replies, indicating that DNS resolution is functioning correctly.

2. Ensure Correct Symlink for /etc/resolv.conf

Ubuntu uses a symbolic link for /etc/resolv.conf managed by systemd-resolved. Let's ensure it's correctly set.

    Check the Symlink:

    bash

ls -l /etc/resolv.conf

Expected Output:

bash

lrwxrwxrwx 1 root root 39 Apr 10 10:00 /etc/resolv.conf -> /run/systemd/resolve/stub-resolv.conf

Recreate the Symlink If Necessary:

If /etc/resolv.conf is not correctly symlinked, recreate it:

bash

sudo rm /etc/resolv.conf
sudo ln -s /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
sudo systemctl restart systemd-resolved

Verify the Symlink:

bash

    cat /etc/resolv.conf

    It should still point to 127.0.0.53 and not contain the manual DNS entries.

3. Flush DNS Cache

To ensure that any stale DNS entries are cleared, flush the DNS cache.

bash

sudo resolvectl flush-caches

4. Check Firewall Settings

Ensure that your firewall isn't blocking DNS traffic (port 53).

    Check UFW Status:

    bash

sudo ufw status

Allow DNS Traffic If Needed:

If UFW is active, allow outgoing DNS traffic:

bash

    sudo ufw allow out 53

    Note: By default, DNS traffic should be allowed, but it's good to verify.

5. Reboot the System

After making these changes, reboot your system to ensure all configurations are properly applied.

bash

sudo reboot

After rebooting, test DNS resolution again:

bash

ping -c 3 google.com

