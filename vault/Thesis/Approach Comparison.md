## Approaches compared, sorted by year
| Name | Loss | Architecture | Year | Approach | Other |  
| ---- | ---- | ------------ | ---- | -------- | ----- |  
| MatchNet | Binary Cross-Entropy | CNN/AlexNet inspired | 2015 | Positive and negative patches | Quantized features for compact representation |
| QuadNetworks | Hinge | CNN | 2016 | Unsupervised Rank-learning | Finds points between image and transformed version of it, which allows to find the transformation at test time |
| DELF | CrossEntropy | CNN+Attention | 2016 | 1. Train CNN classifier 2. Train attentive feature-selecter which chooses most important features from classifier  | Nearest Neighbor search for finding corresponding images |
| LIFT | L2-Norm (Contrastive) | CNN | 2016 | Model consists of detector, orientation estimator and descriptor; trained independently | Hard-Mining used |
| Superpoint | Cross Entropy (Keypoints) + Hinge Loss (Descriptors) | VGG (encoder)  | 2017 | Learn synthesized Keypoints | Both Decoder non-learned |
| AffNet | Hard negative-constant loss | CNN | 2017 | Loss involves not just a similarity optimization but also a discriminative constant to make sure matching works | Separately learn orientation and description, no keypoints |
| D2-Net | Extended triplet (margin ranking) loss | CNN/VGG16 | 2019 | Positive and negative patches | Modified loss for better repeatability of detections |
| SuperGlue | CrossEntropy | GNN | 2019 | Self-attention (intra-image) and Cross-attention (inter-image) | Also let the model allow to assign unmatched keypoints |
| R2D2 |  |  | 2019 |  |  |
| DISK | Reward-based (Reinforcement Learning) | U-Net | 2020 | Positive, neutral or negative reward dependent if match or not | Multiple sampling steps in training |
| D2D |  |  | 2020 |  |  |
| Harris Corner |  |  | 2021 |  |  |
| COTR |  |  | 2021 |  |  |
| LoFTR | BinaryCrossEntropy + L2 (for Regression)  | Feature Pyramid Network (FPN) | 2021 | Detector free feature matching with attention | Coarse matching, then refine matches |
| ASpanFormer |  |  | 2022 |  |  |
| DKM |  |  | 2022 |  |  |
| PosFeat |  |  | 2022 |  |  |
| MatchFormer |  |  | 2022 |  |  |
| LightGlue |  |  | 2023 |  |  |
| CasMTR |  |  | 2023 |  |  |