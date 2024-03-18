- Learn fixed-dimensional representation ([[Annotations/MatchNet#^g0fanxbw6bu]])
	- But: Dont compare these descriptors in feature space but with a learned distance metric
- Feature network maps image-patches to bottleneck. The metric-network classifies two concatenated features binary ([[Annotations/MatchNet#^g8uwloubndd]])
- Simple CNN architecture ([[Annotations/MatchNet#^q11wm2o5s8]])
- Specific way for normalization ([[Annotations/MatchNet#^ewyy62j4lj]])
- Sampling in training: Matching and non-matching pairs are highly unbalanced. The sample allows to generate equal number of positive and negative examples in each mini-batch [[Annotations/MatchNet#^2h3xrecfltq]]
- The buffer size R defines the number of patches used for negative examples. It is a trade-off between memory and negative variety. They used R = 16384 [[Annotations/MatchNet#^twlf1y6wkb]]
- Disasembled feature network and metric network at test time [[Annotations/MatchNet#^pggfaqe9cj]]
- Use false positive rate as evaluation metric (there are a lot of negative-pairs at test time. How many of them are incorrectly classified as positive pairs?). Since the naive optimum for this metric is to classify all pairs as negatives by default, the recall is set to 95% (= 95% of all positives are correctly classified as positives) ([[Annotations/MatchNet#^txjgsc09fjb]])
- 



###  MatchNet Questions
- regarding general siamese networks: can i simply run model(x_1) and model(x_2) and backprop after that?