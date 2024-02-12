- Key idea: Map patches of image A and image B  to a single real-valued response (Both images come from the same source image, but were manipulated with different transformations). Sort these patches wrt the response, one ranking for image A and one ranking for image B. Train the NN (which computes the response) in a way s.t. two corresponding patches from image A and B have the same position in their ranking ([[Annotations/Quad-Networks#^z67pqurhlp]])
- QuadNetwork considers the interest points coming from the top/bottom quantiles of the response function ([[Annotations/Quad-Networks#^wzs64kz0dza]])
- Two corresponding points in the image shall result in the same position in their image-depending ranking. This can be mathematically expressed as here: [[Annotations/Quad-Networks#^sxyk6wst1s]]]
- Loss: To ensure that the ranking-position of two corresponding patches is the same, they always compare two patch-pairs, i.e. 4 patches consisting of 2 corresponding matches. If the sign of the difference of their response is the same, these two matches are ordered in the right way: ([[Annotations/Quad-Networks#^appnmq8r8mu]])
- Train this response function with the hinge loss makes sure response is always > 0: [[Annotations/Quad-Networks#^zxoqajygwj8]]
- 


### QuadNetwork Questions
- If they work patch-wise, how do they find the patches (test time) and how did they create the response image?
- Hard to use for Brain-data, because two corresponding images cant be created using simple transformations.