- Implementations so far used naive way for assigning matches: Replace this, what they call it, "simple matching heuristic" like euclidean distances between descriptors. Learn the matching process from given local features ([[Annotations/SuperGlue#^lt45j95bd1m]])
- SuperGlue is just the Middle-End matcher, ie it is another front-end required which finds local features in the first place
- We can interpret this problem as Graph matching, ie a set of edges without common vertices ([[Annotations/SuperGlue#^s8ibywtn5kb]]). This comes from the physical constraints that a keypoint can have at most one correspondence in the other image and that some keypoints will be unmatched ([[Annotations/SuperGlue#^gnmw4m1u9rt]])
- Integrating more contextual cues can increase the distinctiveness of local features: self-attention (intra-image) and cross-attention (inter-image) allows the model to consider its spatial and visual relationship with other keypoints, eg. self-similar, adjacent or co-occurring ones ([[Annotations/SuperGlue#^sxxojtsntl9]]): Let the features communicate with each other! ([[Annotations/SuperGlue#^f6qo6se0nyq]])
- The inner product of the attentional graph neural network gives us a score, with which we build the score matrix ([[Annotations/SuperGlue#^qgc78oznceo]])
- Also let the model allow to assign unmatched keypoints ([[Annotations/SuperGlue#^eclpdz2jx8b]])
- Normalizing the score matrix, similar to row and column softmax, with the sinkhorn algorithm ([[Annotations/SuperGlue#^nkepx8zztm]])
- Train this in a supervised manner with standard negative loss likelihood as a classification problem, given the correspondences from depth map datasets ([[Annotations/SuperGlue#^qs2zz62x9bt]]) 


## SuperGlue questions
- since the partial assignment matrix is not symmetric, how do they decide which nodes to match?