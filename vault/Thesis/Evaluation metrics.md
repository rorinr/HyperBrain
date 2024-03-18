## Mean average precision (mAP) 
- https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
- [[Object_retrieval_with_large_vocabularies_and_fast_spatial_matching_evaluation_definition.pdf]]
- Different definitions of mean average precision
- Most common: Compute the area under curve (AUC) also average precision for each class
- average the AUC-values for each class = mean average precision
- In the original roc we plot the true positive rate vs the false positive rate. Sometimes (for rare diseases) the false positive rate is replaced with precision.

## AUC and Corner error
  
Reflecting on the conventional approach of employing the Area Under the Curve (AUC) of the corner error for the evaluation of image matching algorithms in deep learning, it becomes evident that this metric, which quantifies the offset of corners between two images (A homography between two planes can be uniquely determined by four pairs of corresponding points, under the condition that no three of them are collinear (i.e., they do not all lie on a single straight line)). post homography application based on matched points, may not adequately represent the efficacy of such approaches. This concern stems from the observation that transformations between images within a dataset could surpass simple perspective alterations, rendering homography insufficient for capturing the complexities of spatial relationships. Homography, being tailored for plane-to-plane transformations, could potentially misrepresent the performance of algorithms that accurately predict points amidst more intricate spatial dynamics. Consequently, it is argued that adopting an alternative evaluation routine would provide a more accurate representation of a deep learning model's capabilities under these conditions. Suggested alternative approaches include:

- Feature Matching Accuracy
- Direct Comparison of Descriptors
- Robust Geometric Metrics for complex transformations
- Recall-Precision Curves
- End-to-End Task Performance evaluation
- Structural Similarity Index (SSIM) or Peak Signal-to-Noise Ratio (PSNR) for image quality assessment

These methodologies potentially offer a more nuanced and comprehensive assessment of an algorithm's performance, particularly in scenarios where image transformations extend beyond mere perspective shifts. Another idea is to use the evaluation of the literature without the homography, ie evaluating the raw predicted matches. Evaluating the error of all predicted matches before applying homography, rather than focusing solely on corner errors post-homography, offers a detailed assessment of an image matching algorithm's raw accuracy. This method allows for a direct and comprehensive evaluation of how well the algorithm identifies corresponding points across images, focusing on the accuracy of individual matches rather than their collective support for a geometric transformation model. It has several key benefits:

- **Comprehensive Evaluation**: It examines the performance across all matches, providing a fuller picture of the algorithm's effectiveness.
- **Direct Comparison**: Enables direct comparison between predicted match locations and ground truth, assessing the algorithm's matching capability without the influence of geometric modeling.
- **Local Accuracy Sensitivity**: Focuses on the local accuracy of matches, useful for identifying and improving algorithm performance in challenging areas.
- **Flexibility**: More adaptable to complex transformations between images that a simple homography may not capture.

However, this approach requires careful consideration of error metrics, potential normalization, outlier handling, and comparative analysis against traditional methods. By offering a more nuanced view of matching accuracy, it complements conventional evaluation metrics and supports a deeper understanding of an algorithm's performance.
Another way to evaluate the predicted matches is the match precision plot of the superglue paper: The precision of the approach given a correctness threshold in pixels.

## Number of matches and their distribution
There isn't a standard, widely-adopted single metric that simultaneously considers both the number of matches and their uniform distribution across the image in the context of image matching or feature detection. However, creating a composite metric that integrates these two aspects is feasible and could serve your evaluation needs effectively. Here's a concept for a custom metric that combines both criteria:
### Composite Match Distribution Metric (CMDM)

The idea behind CMDM is to balance the quantity of matches with their spatial distribution uniformity. This can be achieved by multiplying two components: the match count and a uniformity score. The formula for CMDM could look something like this:

CMDM=N×U

Where:

- N is the total number of matches found by the algorithm, normalizing this number could be beneficial to compare different images or sets.
- U is a uniformity score ranging from 0 to 1, where 1 represents a perfectly uniform distribution of matches across the image, and 0 represents a highly clustered distribution. This score could be derived from methods such as calculating the standard deviation of match counts across subdivided regions of the image, entropy measures, or even more sophisticated methods like the Gini coefficient or the coefficient of variation (CV).
### Calculating Uniformity Score (U)

To calculate U, the image can be divided into a grid of equal-sized cells, and the number of matches in each cell counted. A perfectly uniform distribution would have an equal number of matches in each cell. The uniformity score �U can be calculated using several statistical methods; one approach is to use the inverse of the coefficient of variation of the match counts across all cells:
U=1−σ/μ​

Where σ is the standard deviation of the match counts across the cells, and μ is the mean match count per cell. The subtraction from 1 ensures that a higher uniformity score is better, with 1 being the most uniform.

## NN mAP 
## matching score
## Mean average accuracy