- Use affine transformation (slightly rotate and sheering the image) -> Match if center pixel of patch 1 is in patch 2, does not need to be mutual for the first step (next step could be also use mutual)
- use hydra and lightning
- use noise for more robust features
- try to get ground truth
- Another idea: Use different patchsize for crop 2 (crop is same size in terms of pixels, but patches are greater). This maybe allows less ill-defined patch-correspondences because a patch of image 1 may lay completely in one of the larger patches of image 2. 