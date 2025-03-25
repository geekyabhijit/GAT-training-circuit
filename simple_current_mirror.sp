* Simple Current Mirror SPICE Netlist Example
.include /path/to/models/mosfet.lib

* Power supply
VDD VDD 0 DC 1.8V

* Current mirror circuit
* Reference current
IREF VDD DRAIN_REF DC 100u
* Diode-connected reference transistor
M1 DRAIN_REF DRAIN_REF 0 0 NMOS W=10u L=1u m=1
* Mirror transistor 
M2 OUT DRAIN_REF 0 0 NMOS W=10u L=1u m=1
* Load resistor
RLOAD VDD OUT 1k

* Analysis commands
.op
.DC IREF 50u 150u 10u

.end 