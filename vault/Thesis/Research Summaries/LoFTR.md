- find pixel-wise dense matches at a coarse level. Later refine these ([[Annotations/LoFTR#^jucgkuh0nrr]])
- Using a transformer means to have a global receptive field, which allows more globally semantic claims about physical correspondences of points ([[Annotations/LoFTR#^vm0viwt82dm]])
- Detector free approach: extract dense matches between the two sets of the transformed image (=  reduced to 1/8 of the image dimension by CNN backbone) and later refine them to a subpixel level ([[Annotations/LoFTR#^l9m95dtkarl]])
- detector-dependent methods rely on keypoint detectors to find distinctive interest points: Identify areas of unique features, e.g. corners, which are likely to be repeatable across different views. Detector-dependent methods struggle to detect interest points in areas of low texture or redundant information. detector-free approaches try to learn from the global image context, which allows for example to take relative positions of objects into account ([[Annotations/LoFTR#^gc90ozkizos]])
- Similar to superglue, loftr uses self- and cross-attention for message-passing between two sets of descriptors. But while superglue is dependent on another feature detector, loftr propose a detector-free design ([[Annotations/LoFTR#^ms5px54o3gq]])
- detector-free methods directly produce dense descriptors or feature matches without detection ([[Annotations/LoFTR#^sww3blkmo8]])
- Use a Feature pyramid network as cnn backbone to extract local features and reduce the dimensionality to manage computational costs ([[Annotations/LoFTR#^nig4gi80kan]])
- In the 3. matching module a differentiable matching layer produces the confidence matric P which allows to match on a coarse-level. There are two ways to compute this confidence matrix, either by an optimal transport algorithm like in superglue or by using a dual-softmax ([[Annotations/LoFTR#^64wao818pc]])
- Now select matches by only allow probabilities in P higher than a pre-defined threshold and enforce the mutual nearest neighbor criteria ([[Annotations/LoFTR#^dxl4mxjfr2e]])
- The final loss consists of a coarse-level and fine-level loss. The coarse-level loss is NLL (classify if a pixel or a grid is a match or not). The fine-level loss is l2 over the distance between the predicted position and the ground truth. They also weighted this loss with the total variance in order to optimize the refined position that has low uncertainty ([[Annotations/LoFTR#^pb0wjp31j0i]])
- 



## Questions LoFTR
- In 4. coarse-to-fine module they compute the correlation, what do they mean by that?
- In the same step they compute an expectation after a softmax, whay does this mean?