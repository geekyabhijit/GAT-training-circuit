* Differential Pair SPICE Netlist Example
.include /path/to/models/mosfet.lib

* Power supplies
VDD VDD 0 DC 1.8V
VSS 0 VSS DC 1.8V

* Bias current source
IBIAS TAIL VSS DC 100u

* Differential pair transistors
M1 OUT1 IN1 TAIL VSS NMOS W=20u L=1u m=1
M2 OUT2 IN2 TAIL VSS NMOS W=20u L=1u m=1

* Load resistors
RL1 VDD OUT1 10k
RL2 VDD OUT2 10k

* Input voltages
VIN1 IN1 0 DC 0.9V
VIN2 IN2 0 DC 0.9V

* Analysis commands
.op
.DC VIN2 0.8V 1.0V 10m

.end 