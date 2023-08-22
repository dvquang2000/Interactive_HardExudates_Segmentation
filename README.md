# Interactive HardExudates Segmentation

This repository includes the necessary code and configuration files to replicate our work on HardExudates segmentation.

## System requirements
The source code is intended to run on a Linux workstation equipped with a GPU card.

## Acknowledgements
This repository utilizes code from [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation)

## Setting up an environment
This framework is built using Python 3.6 and relies on the PyTorch 1.4.0+. The following command installs all necessary packages:
```.bash
pip3 install -r requirements.txt
```
## Segmentation Demo
We will provide instructions for performing interactive HardExudates segmentation using the combined automatic segmentation and RITM.

You can watch a demo video by following this [link](https://drive.google.com/file/d/1mmMp44DxQ-tn-zi3Hl5j83r64w5Zkxjz/view?usp=drive_link).

To run the interactive HardExudates segmentation demo, do the following steps:

### Step 1: Download weights 
You can download the automatic and interactive segmentation checkpoints for HardEXudates in this [link](https://drive.google.com/drive/folders/1PsjquPLz_dwBmv3_t8WyloqoofesLtb7?usp=drive_link).

Push the automatic segmentation checkpoint into [automatic_weights](automatic_weights) and the interactive segmentation checkpoint into [weights](weights)

### Step2: Run Inference
Execute the following command:
```.bash
python demo.py --checkpoint=RITM_EX.pth --automatic_weight=automatic_weights/VGGUnet_EX.tar --gpu=0
```