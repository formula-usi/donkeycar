# Stay on Track: Phäenomena Exhibition by Università della Svizzera Italiana

### Credits to: Roberto Minelli & Samuele Pasini

This is the guide for the daily usage of the Piracer Car for the Stay on Track: Phäenomena Exhibition by Università della Svizzera Italia.


## Troubleshooting

### Failed to connect the laptop to the Piracer

First, check if an IP Address is shown on the OLED Display of the Piracer Car.
If not, you should connect the Piracer Car to a monitor and a keyboard, and login using the following credentials:
```
formulausi
phaenomena2026!
```
When the OS is running, you should re-initialize the Wi-Fi connection, if it works, an IP will be present on the display.
If the connection works but the IP is not shown, check that an IP is assigned using 
```
ifconfig
```
If it works, you shoule re-install the OLED Software:

```
git clone https://github.com/formula-usi/waveshare-pi-display.git
cd waveshare-pi-display
sudo ./install.sh
```
On the display you should see the same IP that you can get using:

```
ifconfig
```
After the installation, feel free to remove the folder `waveshare-pi-display`.

If the IP works but you still have SSH connection problems, check on your laptop if you are connected to the same Wi-Fi Network, and check if you have the updated IP address on your configuration.

```
Host piracer
        HostName IP_ADDRESS
        User formulausi
        LocalForward 8887 127.0.0.1:8887
        LocalForward 8886 127.0.0.1:8886
        IdentityFile ~/.ssh/id_rsa_piracer
```

If the IP is correct, it is possbile that the link between SSH keys is corrupted, you can create a new SSH key:

```
ssh-keygen
```

Then, when it is asked, save it as 

```
/root/.ssh/id_rsa_piracer_new
```
Then you should copy the key to the Piracer Car using the IP address you can see on the display

```
ssh-copy-id -i ~/.ssh/id_rsa_piracer_new formulausi@IP_ADDRESS
```

When asked, use the password phaenomena2026!


Finally you can set the SSH credentials in the file ~/.ssh/config

```
Host piracer
        HostName IP_ADDRESS
        User formulausi
        LocalForward 8887 127.0.0.1:8887
        LocalForward 8886 127.0.0.1:8886
        IdentityFile ~/.ssh/id_rsa_piracer_new
```