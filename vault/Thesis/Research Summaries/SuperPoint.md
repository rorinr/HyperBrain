- Convolutional net that works on full-sized images -> Computes keypoints and descriptors in one forward pass ([[Annotations/SuperPoint#^f5e4lriah6f]])
- ![[Annotations/SuperPoint#^15mifke3nke]]
- The notion of interest point detection is semantically ill-defined ([[Annotations/SuperPoint#^luwvkqtpcp]])
- They used a synthetic dataset for pretraining. The dataset consists of simple geometric shapes with no ambiguity in the keypoint-locations ([[Annotations/SuperPoint#^kpm9bd45md]])
- The encoder maps the greyscale image to lower dimensionality. 2 decoder, one for interest points (HxWx1) and once of description (HxWxD, follow ([[Annotations/SuperPoint#^jrfz02k4ta8]]) 
- The interest-point decoder has latent dimension of 8x8x65, where the 65th channel is an additional "no interest point"-dustbin. The use this tensor as input for the softmax to find keypoints (since softmax is for classification, on interpretation is to classify if a pixel is a keypoint or not. What i dont understand: Which pixel are compared? One for each 8x8 cell, s.t. we end up with one keypoint per 8x8 cell? This would make sense -> The dustbin gets removed so it is not required to have exactly one keypoint per cell, some have none. The reshaping process seems a bit unclear through pixel shuffle) ([[Annotations/SuperPoint#^r5jyvm4gtta]])
- The descriptor decoder maps the output of the encoder to WxHxD via bicubic interpolation and l2-norm ([[Annotations/SuperPoint#^z7ye5ri89n]])
- The first part of the loss is the interest point detector loss, which is simply cross-entropy over the 8x8 pixel cells + the additional 65th pixel, ie dustbin. If two interest points land in same bin, select on randomly. Run this loss for two versions of the same image, ie. one is manipulated ([[Annotations/SuperPoint#^opmov2dn4e]])
- Descriptor loss is applied pairwise to corresponding cells of the first and second image. They used a threshold of 8, ie two cells are corresponding if the center pixel is 8 or less pixel away ([[Annotations/SuperPoint#^n4rjrcq0op]])
- The transformed each cell to a vector and computed the dot product between all vectors from first image to second image. The loss is minimal if positive pairs end up with a dot product of m_p or higher and negative pairs with dot product of m_n or less ([[Annotations/SuperPoint#^78k5dhr8a34]])
- For images in target domain they introduce a process called homographic adaptation: The already trained base detector (trained on synthesized images) is now capable to detect interest points. They managed to produce pseudo-ground-truth points by applying the base detector to multiple versions of the same unlabeled image. The combination of all found interest points is now the supervision/label for the before unlabeled image ([[Annotations/SuperPoint#^1nijx1gjz8g]])
- The initial base detector struggled to find repeatable detections (only trained on the synthesized dataset). But after further homographic training, the detector improved significantly ([[Annotations/SuperPoint#^kx8fyiyhjli]])



### Questions SuperPoint
- What is the output exactly and how do they determine if a pixel is a keypoint?