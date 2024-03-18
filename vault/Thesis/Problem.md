## Overview

^b14aa9

Find a proper method for stacking multiple consecutive scans of brain slices from a single human brain on top of each other, resulting in a 3-dimensional model of the complete brain (based on n 2-dimensional scans). 
This mapping cannot be linear since brain slices are extremely thin and non-static. These brain slices can be imagined like cling film due to their thin, flexible, and unpredictable nature. Dynamic structural changes occur in brain slices, similar to how cling film deforms and changes shape when handled. This dynamic behavior causes variation in the 2D scans of the same brain section when imaged multiple times because the deformations of the slices can result in differing appearances with each scan. Even when attempting parallel placement of the same brain slice on the underlying surface in multiple scans, the dynamic nature of the slice can cause subtle shifts, bends, or distortions, leading to differences.
There is a non-published method from the chair that solves this problem. However, it needs already roughly coordinated images: It needs some and above all well distributed points belonging together from image 1 to image 2. The task of the thesis is to solve exactly this problem, e.g. using contrastive learning.

## State of research
The above problem is also known as image matching: Finding a sufficient number of pixel or region correspondences between two images of the same scene to estimate the geometric relationship between the cameras that produced those images (https://paperswithcode.com/task/image-matching).
Typically, the data are natural images of the same object, but taken from different perspectives. The problem at hand also requires linking related regions of the images, but with the following difference: We are dealing with medical images taken from the same perspective, but from different (though similar) objects.
One problem in current research is that different paper provide different evaluation metrics which complicates comparability. 

## Challenges
Unsupervised: Articles read so far which introduced unsupervised approaches for image matching (or similar tasks) make use of image augmentation-methods. This works well for natural images and their changing perspectives in real life for the following reason: Image augmentations like random cropping or affine transformation are very similar to the changes under different perspectives (like in )
