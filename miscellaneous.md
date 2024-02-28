- Projective transformation instead of affine

- refactor evaluation? 
- handle exceptions when perspective sampling is out of boundaries, max_y < min_y :(
- going through loftr check if i missed anything
- print hist of parameters similar to gradients
- use superpoint augmentations
- rewrite refinement supervision so that its exact?
- torch compile
- torch gradient accumulation
- torch half precision

- use same channels as loftr
- dustbin to allow model to say there is no match for this patch? check if loftr used it
- gradient clipping if graidents to big (double check this before implementing)

- have look at relative coordinates: Does it allow floats output? Should be to be pixel precise!

- learnig rate schedule


- compute loftr prediction again with more comparibitilty (same padding etc.)

- it seems like there are a lot of predicted oarse matches that are almost true, ie they are lying one patch off to the right one. Maybe its reasonable to plot the wrong matches with some color thats indicating how much the prediction is wrong

- Train a model only on one image and find out it the performance (regarding loss and evaluation metrics) is worse than training on 8 images. This will tell us if more data and compute will yield in better results. Make in dataset a parameter s.t. you can choose which images should be part of train and test set --> Use new evaluation method

- Train self-supervised (and maybe even evaluate self-supervised). This will tell us if labeled data are necessary at all - and since the supervision is noisy the performance could be even better (but we wouldnt to detect better performance since we dont have the needed ground truth to tell so)  --> Use new evalutation routine

- Find an example where a projective transformation is not enough 
    - Use the evaluation images and their deformation to compute a homography. Now show that this homography doesnt yield in pixel to pixel correspondences

- https://theaisummer.com/weights-and-biases-tutorial/ use this for training

- maybe after self.layer3_outconv1 another batchnorm before passing it further (gradients get pretty small here)

- Future ideas for better performance:
    - Add some noise (in both images?) for make features more stable as pawel recommended
    - 1/8 and 1/2
    - vanilla attention
    - another cross entropy instead of l2
    - switch crop_1 and crop_2 so that the first one is transformed
    - final layer for relative coordinates tanh?
    - original loftr doesnt take the sqrt in fine loss
    - clamp in fine matching to make sure predicted relative coordinates are 0.5 max
    - outconvs have pretty small gradient. Find out why (it is the same implementation as in loftr?)
    - use downscale factor 5 instead of 10


- read the deep learning rules pawel sent 
 
- Fully understand the attention mechanism and the used transformer

- Ransac per patch to exclude outlier