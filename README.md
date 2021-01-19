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
   
## Performance
Table: RMSE per camera property, ten runs
| n | Iterations | Roll degree | Tilt degree | Elevation in m|
|---|---|---|---|---|
| 1 | 1e4 | 56.23 | 20.91 | 1.02
| 2 | 1e4 | 54.97 | 22.40 | 1.01
| 3 | 1e4 | **47.68** |  **18.59** | 1.00 
| 4 | 1e4 | 52.01 | 23.76 | 1.01 
| 5 | 1e4 | 50.09 | 21.32 | 0.99  
| 6 | 1e4 | 48.51 | 20.06 | 0.98 
| 7 | 1e4 | 51.90 | 21.73 | 0.98
| 8 | 1e4 | 47.26 | 22.09 | 0.95
| 9 | 1e4 | 50.51 | 22.29 | 1.01
| 10 | 1e4 | 51.74 | 20.88 |  **0.96** 
| mean |  | 51.09 | 21.40 | 0.99 
| std |  | 2.94 | 1.42 | 0.02

Plot with results:
![](notebooks\stats_props_partial.png)

## More about EU-Project ASSISTANCE
**A**dapted Situation Awarene**SS** tools and ta**I**lored training scenarios for increa**S**ing capabiliTies and enh**AN**cing the prote**C**tion of First Respond**E**rs – **ASSISTANCE** is an international research project funded by the European Commission under the **Horizon 2020** programme in Secure Societies Challenge addressing the **SU-DRS02-2018-2019-2020 (Technologies for first responders) topic**.

The main purpose of ASSISTANCE project is twofold: **to help and protect different kind of first responders’ (FR) organizations that work together during the mitigation of large disasters (natural or man-made) and to enhance their capabilities and skills for facing complex situations** related to different types of incidents.

For more, see: https://assistance-project.eu/
