# Online Collaboration-based Visual Tracking for UAV with Spatial-to-Semantic Information and Multi-Recommender Voting 
Matlab implementation of Online Collaboration-based Visual Tracking for UAV with
Spatial-to-Semantic Information and Multi-Recommender Voting (SIMV) tracker.

## Description and Instructions

### Configuration

1. Download VGG-Net-19 by cliking [here](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it in `/model`.
2. Download matconvnet toolbox [here](http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz) and put it directly in root (which means there should be a `matconvnet/` folder in SIMV folder.

### How to run

* runTracker.m  -  set the directory for UAV123 images sequences 

  ​                          -  choose UAV123 image sequences for the SIMV tracker.

## Acknowledgements

 This work borrowed the framework from the MCCT tracker (https://github.com/594422814/MCCT) and the parameter settings from DSST (http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html).


