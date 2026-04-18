# Demo Instructions

## What the demo shows
We running a trained CKAN model on an FPGA board (PYNQ-Z2) that inference 500 images from  MNIST and Fashion-MNIST dataset.

## What you need (all is provided in the demo_inputs folder)
- PYNQ-Z2 board powered on and connected to laptop via USB
- The bitstream file ready (`Ckan_model.bit`)
- A few test images from the dataset converted to hexadecimal
- axi_drivers files to transfer .bit file to fpga and use for inference 

## Steps

**1. Connect the board**
- Plug the USB cable from the board to the laptop
- Open the terminal and use command :"sudo minicom -D /dev/ttyUSB0 -b 115200" to connect to the terminal of the pynq board
- find the ip of the board using "ifconfig"
- use the ip to open jupyter notebook provide for inference in the browser 


All the finding and results are given in the Jupyter notebook only. 
