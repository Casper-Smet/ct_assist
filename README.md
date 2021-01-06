# CameraTransform-Assistance
`ct_assist`, or CameraTransform-Assistance, is a library for finding head-feet pairs on images. This allows for further automation of the [CameraTransform](https://github.com/rgerum/cameratransform). It extracts head-feet pairs from a binary mask generated using instance segmentation. This requires the user to input a pytorch tensor.

This project was made for EU-Project ASSISTANCE. See below for more.

## Use case
This library will be applied to images of fluid spills in order to estimate their area and release rate. See [fluid_estimator](https://github.com/Casper-Smet/spill_estimator), request access from the [the author](mailto:casper.smet@gmail.com).


## CameraTransform
This repo extends [CameraTransform](https://github.com/rgerum/cameratransform)'s functionality by trying to find the reference objects with Computer Vision. 

## Installing dependencies
We provide a Anaconda environment file `[env.yaml]` for installing this repo's dependencies. 


1. `git clone https://github.com/Casper-Smet/ct_assist.git`
2. `cd .\ct_assist\`
3. `conda env create -f env.yml`
4. `conda activate ct_assist`
5. `pip install .`
   

## More about EU-Project ASSISTANCE
**A**dapted Situation Awarene**SS** tools and ta**I**lored training scenarios for increa**S**ing capabiliTies and enh**AN**cing the prote**C**tion of First Respond**E**rs – **ASSISTANCE** is an international research project funded by the European Commission under the **Horizon 2020** programme in Secure Societies Challenge addressing the **SU-DRS02-2018-2019-2020 (Technologies for first responders) topic**.

The main purpose of ASSISTANCE project is twofold: **to help and protect different kind of first responders’ (FR) organizations that work together during the mitigation of large disasters (natural or man-made) and to enhance their capabilities and skills for facing complex situations** related to different types of incidents.

For more, see: https://assistance-project.eu/
