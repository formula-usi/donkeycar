# Stay on Track: Phäenomena Exhibition by Università della Svizzera Italiana

### Credits to: Roberto Minelli & Samuele Pasini

This is the guide starting from Zero to setup and use the Waveshare Piracer Car for the "Stay on Track" Exhibition. The target of this guide are the managers of the exhibition, a specific guide for daily usage is available in the page [Daily](docs/daily.md).
The original donkeycar README file is available in the page [Donkeycar](docs/donkeycar.md)

This guide will be structured as follows: The first part will cover the [Installation and Initialization](#installation-and-initialization), setup the [Path-ological](#path-ological) application to manage the circuits, how to perform a [Driving Session](#driving-session), a [Simulator Session](#simulator-session), [Model Training](#model-training), run the Piracer Car with [Autonomous Driving](#autonomous-driving).

## Installation and Initialization

### Raspberry OS Installation
The first step is the installation of the OS on the SD Card. Insert the SD Card in an SD Card reader, connect to you laptop and follow the steps to install the Raspberry PI OS following the steps at [this link](http://diyrobocars.com/2025/06/28/tips-for-installing-donkeycar-on-the-waveshare-piracer-pro/).

Use the following credentials:
```
formulausi
phaenomena2026!
```

### Enable SSH
- Enable `SSH` in `Services`
    - Use **Password Authentication**, otherwise you can only login with RSA keys.
    - If you forget to do so, [you can fix it](https://www.bodhost.com/kb/how-to-enable-ssh-password-authentication/) later.
- Finish by applying OS customization settings!

### Assembly
Once the installation is completed, remove the SD card from your laptopt and insert it into the the Raspberry.
You can now assembly the Piracer Car following the [Official Guide](https://www.waveshare.com/wiki/File:Piracer_pro_ai_kit-en2.pdf).


### Connect to Wi-Fi
For the first time, you should already have a Wi-Fi connection configures during installation, if you want to change it you should connect the car to the monitor and they keyboard, then login with the credentials used in the installation, and select the wifi network to use, the default one used as hotspot during tests is:
```
formulausi
StoccoBoy!
```

### OLED Display initialization

Use command belows to install service for OLED display that displays IP address, battery status, etc.
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



### SSH Connection from the laptop
First you should create the SSH Key:

```
ssh-keygen
```

Then, when it is asked, save it as 

```
/root/.ssh/id_rsa_piracer
```
Then you should copy the key to the Piracer Car using the IP address you can see on the display

```
ssh-copy-id -i ~/.ssh/id_rsa_piracer formulausi@IP_ADDRESS
```

When asked, use the password phaenomena2026!


Finally you can set the SSH credentials in the file ~/.ssh/config

```
Host piracer
        HostName IP_ADDRESS
        User formulausi
        LocalForward 8887 127.0.0.1:8887
        LocalForward 8886 127.0.0.1:8886
        IdentityFile ~/.ssh/id_rsa_piracer
```
Be careful, the IP Address can change during the days, if you see a different IP Address on the display, remember to change the configuration.

### Bluetooth Initialization

In the SSH Session, TODO


### Create a Donkey Car Application
In the SSH Session, use this command:
```
donkey createcar --path ~/piracerpro
```

### Update Car Configuration
- Navigate to the Donkey Car application created in the previous step (e.g., `cd ~/piracerpro`)
- Replace the file `myconfig.py` with the following configurations:

```

```
If, after running the car, you notice that there are calibration problems, you can follow the next section to change the obtain more precise parameters for the configuration

### Calibrate the Car
Follow [this guide](https://docs.donkeycar.com/guide/calibrate/) to calibrate steering and throttle.

### Conect the Car to the NVIDIA Spark

An Nvidia Spark (aka Gold) will be used to train the models, you should enstablish an SSH Connection between the car and the Gold.

First you should create the SSH Key:

```
ssh-keygen
```

Then, when it is asked, save it as 

```
/root/.ssh/id_rsa_gold
```
Then you should copy the key to the NVIDIA Spark

TODO ssh copy id

Finally you can set the SSH credentials in the file ~/.ssh/config

```
Host gold 
    HostName gold.si.usi.ch
    User formulausi
    Port 222
    IdentityFile ~/.ssh/id_rsa_gold
    LocalForward 11000 127.0.0.1:11000
```
