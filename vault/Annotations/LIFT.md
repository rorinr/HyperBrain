annotation-target::LIFT.pdf

>%%
>```annotation-json
>{"created":"2023-10-07T10:43:26.215Z","text":"One patch only contains one keypoint","updated":"2023-10-07T10:43:26.215Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":10556,"end":10713},{"type":"TextQuoteSelector","exact":"assume they contain only one dominant local feature at the given scale, whichreduces the learning process to finding the most distinctive point in the patch.","prefix":"ed Invariant Feature Transform 5","suffix":"To train our network we create t"}]}]}
>```
>%%
>*%%PREFIX%%ed Invariant Feature Transform 5%%HIGHLIGHT%% ==assume they contain only one dominant local feature at the given scale, whichreduces the learning process to finding the most distinctive point in the patch.== %%POSTFIX%%To train our network we create t*
>%%LINK%%[[#^d3nf1x1vxwg|show annotation]]
>%%COMMENT%%
>One patch only contains one keypoint
>%%TAGS%%
>
^d3nf1x1vxwg


>%%
>```annotation-json
>{"created":"2023-10-07T10:48:33.072Z","text":"4 patches: 2 matching patches showing the same 3d point, 1 patch which show another 3d point and 1 patch which is used as negative example and does not contain any distinctive feature point.\nNote: The approach assumes that the data is labeled, ie that is known where good keypoints are.","updated":"2023-10-07T10:48:33.072Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":10713,"end":11294},{"type":"TextQuoteSelector","exact":"To train our network we create the four-branch Siamese architecture picturedin Fig. 2. Each branch contains three distinct CNNs, a Detector, an OrientationEstimator, and a Descriptor. For training purposes, we use quadruplets of imagepatches. Each one includes two image patches P1 and P2, that correspond todifferent views of the same 3D point, one image patch P3, that contains the pro-jection of a different 3D point, and one image patch P4 that does not contain anydistinctive feature point. During training, the i-th patch Pi of each quadrupletwill go through the i-th branch.","prefix":" distinctive point in the patch.","suffix":"To achieve end-to-end differenti"}]}]}
>```
>%%
>*%%PREFIX%%distinctive point in the patch.%%HIGHLIGHT%% ==To train our network we create the four-branch Siamese architecture picturedin Fig. 2. Each branch contains three distinct CNNs, a Detector, an OrientationEstimator, and a Descriptor. For training purposes, we use quadruplets of imagepatches. Each one includes two image patches P1 and P2, that correspond todifferent views of the same 3D point, one image patch P3, that contains the pro-jection of a different 3D point, and one image patch P4 that does not contain anydistinctive feature point. During training, the i-th patch Pi of each quadrupletwill go through the i-th branch.== %%POSTFIX%%To achieve end-to-end differenti*
>%%LINK%%[[#^mwg0axgvtu|show annotation]]
>%%COMMENT%%
>4 patches: 2 matching patches showing the same 3d point, 1 patch which show another 3d point and 1 patch which is used as negative example and does not contain any distinctive feature point.
>Note: The approach assumes that the data is labeled, ie that is known where good keypoints are.
>%%TAGS%%
>
^mwg0axgvtu


>%%
>```annotation-json
>{"created":"2023-10-07T11:39:37.343Z","updated":"2023-10-07T11:39:37.343Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":11389,"end":11979},{"type":"TextQuoteSelector","exact":"1. Given an input image patch P, the Detector provides a score map S.2. We perform a soft argmax [12] on the score map S and return the locationx of a single potential feature point.3. We extract a smaller patch p centered on x with the Spatial Transformerlayer Crop (Fig. 2). This serves as the input to the Orientation Estimator.4. The Orientation Estimator predicts a patch orientation θ.5. We rotate p according to this orientation using a second Spatial Transformerlayer, labeled as Rot in Fig. 2, to produce pθ.6. pθ is fed to the Descriptor network, which computes a feature vector d","prefix":" branch areconnected as follows:","suffix":".Note that the Spatial Transform"}]}]}
>```
>%%
>*%%PREFIX%%branch areconnected as follows:%%HIGHLIGHT%% ==1. Given an input image patch P, the Detector provides a score map S.2. We perform a soft argmax [12] on the score map S and return the locationx of a single potential feature point.3. We extract a smaller patch p centered on x with the Spatial Transformerlayer Crop (Fig. 2). This serves as the input to the Orientation Estimator.4. The Orientation Estimator predicts a patch orientation θ.5. We rotate p according to this orientation using a second Spatial Transformerlayer, labeled as Rot in Fig. 2, to produce pθ.6. pθ is fed to the Descriptor network, which computes a feature vector d== %%POSTFIX%%.Note that the Spatial Transform*
>%%LINK%%[[#^kfb60lmsupq|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kfb60lmsupq


>%%
>```annotation-json
>{"created":"2023-10-07T11:39:57.086Z","text":"Spatial transformers are not learned modules","updated":"2023-10-07T11:39:57.086Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":12102,"end":12130},{"type":"TextQuoteSelector","exact":"They are not learned modules","prefix":"e preserving differentiability. ","suffix":".Also, both the location x propo"}]}]}
>```
>%%
>*%%PREFIX%%e preserving differentiability.%%HIGHLIGHT%% ==They are not learned modules== %%POSTFIX%%.Also, both the location x propo*
>%%LINK%%[[#^8rojm8wx4a7|show annotation]]
>%%COMMENT%%
>Spatial transformers are not learned modules
>%%TAGS%%
>
^8rojm8wx4a7


>%%
>```annotation-json
>{"created":"2023-10-07T11:44:04.053Z","text":"Common problem: learn the different tasks separately usually yields in better resutls","updated":"2023-10-07T11:44:04.053Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":12460,"end":12543},{"type":"TextQuoteSelector","exact":"Our early attempts at training the network as a wholefrom scratch were unsuccessful","prefix":"ningthe weights is non-trivial. ","suffix":". We therefore designed a proble"}]}]}
>```
>%%
>*%%PREFIX%%ningthe weights is non-trivial.%%HIGHLIGHT%% ==Our early attempts at training the network as a wholefrom scratch were unsuccessful== %%POSTFIX%%. We therefore designed a proble*
>%%LINK%%[[#^0cr76r5jl0b7|show annotation]]
>%%COMMENT%%
>Common problem: learn the different tasks separately usually yields in better resutls
>%%TAGS%%
>
^0cr76r5jl0b7


>%%
>```annotation-json
>{"created":"2023-10-07T14:00:51.348Z","text":"AFAIK: There are e.g. 59k points in the 3d scene, where each point has multiple observation, ie it appears in multiple image.","updated":"2023-10-07T14:00:51.348Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":13809,"end":14229},{"type":"TextQuoteSelector","exact":"We used the collections from PiccadillyCircus in London and the Roman Forum in Rome from [29] to reconstruct the3D using VisualSFM [30], which relies of SIFT features. Piccadilly contains 3384images, and the reconstruction has 59k unique points with an average of 6.5 ob-servations for each. Roman-Forum contains 1658 images and 51k unique points,with an average of 5.2 observations for each. Fig. 3 shows some examples.","prefix":"ed to photo-tourism image sets. ","suffix":"We split the data into training "}]}]}
>```
>%%
>*%%PREFIX%%ed to photo-tourism image sets.%%HIGHLIGHT%% ==We used the collections from PiccadillyCircus in London and the Roman Forum in Rome from [29] to reconstruct the3D using VisualSFM [30], which relies of SIFT features. Piccadilly contains 3384images, and the reconstruction has 59k unique points with an average of 6.5 ob-servations for each. Roman-Forum contains 1658 images and 51k unique points,with an average of 5.2 observations for each. Fig. 3 shows some examples.== %%POSTFIX%%We split the data into training*
>%%LINK%%[[#^soagk012mwi|show annotation]]
>%%COMMENT%%
>AFAIK: There are e.g. 59k points in the 3d scene, where each point has multiple observation, ie it appears in multiple image.
>%%TAGS%%
>
^soagk012mwi


>%%
>```annotation-json
>{"created":"2023-10-07T14:03:34.343Z","updated":"2023-10-07T14:03:34.343Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":14357,"end":14474},{"type":"TextQuoteSelector","exact":"To build the positive trainingsamples we consider only the feature points that survive the SfM reconstructionprocess.","prefix":" validation set and vice-versa. ","suffix":" To extract patches that do not "}]}]}
>```
>%%
>*%%PREFIX%%validation set and vice-versa.%%HIGHLIGHT%% ==To build the positive trainingsamples we consider only the feature points that survive the SfM reconstructionprocess.== %%POSTFIX%%To extract patches that do not*
>%%LINK%%[[#^9z3sw4t9zat|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^9z3sw4t9zat


>%%
>```annotation-json
>{"created":"2023-10-07T14:10:19.419Z","text":"two descripted embeddings should have min. euclidean distance  if corresponding to the same physical 3d point and max. (or C) distance if not. ","updated":"2023-10-07T14:10:19.419Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":16354,"end":16957},{"type":"TextQuoteSelector","exact":"We train the Descriptor by minimizing the sum of the loss for pairs of cor-responding patches (p1θ,p2θ) and the loss for pairs of non-corresponding patches(p1θ,p3θ). The loss for pair (pkθ,plθ) is defined as the hinge embedding loss of theEuclidean distance between their description vectors. We writeLdesc(pkθ,plθ) ={∥∥hρ(pkθ)−hρ(plθ)∥∥2 for positive pairs, andmax (0,C −∥∥hρ(pkθ)−hρ(plθ)∥∥2) for negative pairs ,(2)where positive and negative samples are pairs of patches that do or do notcorrespond to the same physical 3D points, ‖·‖2 is the Euclidean distance, andC = 4 is the margin for embedding.","prefix":"SfM togenerate image patches pθ.","suffix":"We use hard mining during traini"}]}]}
>```
>%%
>*%%PREFIX%%SfM togenerate image patches pθ.%%HIGHLIGHT%% ==We train the Descriptor by minimizing the sum of the loss for pairs of cor-responding patches (p1θ,p2θ) and the loss for pairs of non-corresponding patches(p1θ,p3θ). The loss for pair (pkθ,plθ) is defined as the hinge embedding loss of theEuclidean distance between their description vectors. We writeLdesc(pkθ,plθ) ={∥∥hρ(pkθ)−hρ(plθ)∥∥2 for positive pairs, andmax (0,C −∥∥hρ(pkθ)−hρ(plθ)∥∥2) for negative pairs ,(2)where positive and negative samples are pairs of patches that do or do notcorrespond to the same physical 3D points, ‖·‖2 is the Euclidean distance, andC = 4 is the margin for embedding.== %%POSTFIX%%We use hard mining during traini*
>%%LINK%%[[#^uught21bjy9|show annotation]]
>%%COMMENT%%
>two descripted embeddings should have min. euclidean distance  if corresponding to the same physical 3d point and max. (or C) distance if not. 
>%%TAGS%%
>
^uught21bjy9


>%%
>```annotation-json
>{"text":"They used hard-mining, which means that they only backpropagate the datapoints with high loss. the mining ratio r defines how many of the forward-propagated datapoints to use. r=1 means to use all of them, r=2 half of them (the half with the higher losses) and so on","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":16957,"end":17314},{"type":"TextQuoteSelector","exact":"We use hard mining during training, which was shown in [10] to be critical fordescriptor performance. Following this methodology, we forward Kf sample pairsand use only the Kb pairs with the highest training loss for back-propagation,where r = Kf/Kb ≥1 is the ‘mining ratio’. In [10] the network was pre-trainedwithout mining and then fine-tuned with r = 8.","prefix":"= 4 is the margin for embedding.","suffix":"Here, we use an increasingminin"}]}],"created":"2023-10-07T14:24:24.622Z","updated":"2023-10-07T14:24:24.622Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf"}
>```
>%%
>*%%PREFIX%%= 4 is the margin for embedding.%%HIGHLIGHT%% ==We use hard mining during training, which was shown in [10] to be critical fordescriptor performance. Following this methodology, we forward Kf sample pairsand use only the Kb pairs with the highest training loss for back-propagation,where r = Kf/Kb ≥1 is the ‘mining ratio’. In [10] the network was pre-trainedwithout mining and then fine-tuned with r = 8.== %%POSTFIX%%Here, we use an increasingminin*
>%%LINK%%[[#^h08bjite7ji|show annotation]]
>%%COMMENT%%
>They used hard-mining, which means that they only backpropagate the datapoints with high loss. the mining ratio r defines how many of the forward-propagated datapoints to use. r=1 means to use all of them, r=2 half of them (the half with the higher losses) and so on
>%%TAGS%%
>#question
^h08bjite7ji


>%%
>```annotation-json
>{"created":"2023-10-07T14:41:01.513Z","text":"Find the rotation angel in the first place and then rotate it","updated":"2023-10-07T14:41:01.513Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":17986,"end":18478},{"type":"TextQuoteSelector","exact":"We therefore propose to use Spatial Transformers [11] instead to learn theorientations. Given a patch p from the region proposed by the detector, theOrientation Estimator predicts an orientationθ = gφ(p) , (3)where g denotes the Orientation Estimator CNN, and φ its parameters.Together with the location x from the Detector and P the original imagepatch, θ is then used by the second Spatial Transformer Layer Rot(.) to providea patch pθ = Rot (P,x,θ), which is the rotated version of patch p","prefix":"compute the description vectors.","suffix":".We train the Orientation Estima"}]}]}
>```
>%%
>*%%PREFIX%%compute the description vectors.%%HIGHLIGHT%% ==We therefore propose to use Spatial Transformers [11] instead to learn theorientations. Given a patch p from the region proposed by the detector, theOrientation Estimator predicts an orientationθ = gφ(p) , (3)where g denotes the Orientation Estimator CNN, and φ its parameters.Together with the location x from the Detector and P the original imagepatch, θ is then used by the second Spatial Transformer Layer Rot(.) to providea patch pθ = Rot (P,x,θ), which is the rotated version of patch p== %%POSTFIX%%.We train the Orientation Estima*
>%%LINK%%[[#^man5ibu1fgm|show annotation]]
>%%COMMENT%%
>Find the rotation angel in the first place and then rotate it
>%%TAGS%%
>
^man5ibu1fgm


>%%
>```annotation-json
>{"created":"2023-10-09T12:52:17.207Z","text":"Problem definition","updated":"2023-10-09T12:52:17.207Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":1373,"end":1704},{"type":"TextQuoteSelector","exact":"These new algorithms, however, address only a single step in the completeprocessing chain, which includes detecting the features, computing their orienta-tion, and extracting robust representations that allow us to match them acrossimages. In this paper we introduce a novel Deep architecture that performs allthree steps together.","prefix":"raditional methods [6,7,8,9,10].","suffix":" We demonstrate that it achieves"}]}]}
>```
>%%
>*%%PREFIX%%raditional methods [6,7,8,9,10].%%HIGHLIGHT%% ==These new algorithms, however, address only a single step in the completeprocessing chain, which includes detecting the features, computing their orienta-tion, and extracting robust representations that allow us to match them acrossimages. In this paper we introduce a novel Deep architecture that performs allthree steps together.== %%POSTFIX%%We demonstrate that it achieves*
>%%LINK%%[[#^19htf4xtbmy|show annotation]]
>%%COMMENT%%
>Problem definition
>%%TAGS%%
>
^19htf4xtbmy


>%%
>```annotation-json
>{"created":"2023-10-09T13:14:46.053Z","text":"The SfM-algorithm finds corresponding points in 3d, s.t. we know which image patches of 2 different images of the same scene showing the same point in 3d. ","updated":"2023-10-09T13:14:46.053Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":3379,"end":3609},{"type":"TextQuoteSelector","exact":" we build a Siamese network and train it using the feature points producedby a Structure-from-Motion (SfM) algorithm that we ran on images of a scenecaptured under different viewpoints and lighting conditions, to learn its weights","prefix":"an effective manner. To thisend,","suffix":".We formulate this training prob"}]}]}
>```
>%%
>*%%PREFIX%%an effective manner. To thisend,%%HIGHLIGHT%% ==we build a Siamese network and train it using the feature points producedby a Structure-from-Motion (SfM) algorithm that we ran on images of a scenecaptured under different viewpoints and lighting conditions, to learn its weights== %%POSTFIX%%.We formulate this training prob*
>%%LINK%%[[#^4rmfdfvp8ba|show annotation]]
>%%COMMENT%%
>The SfM-algorithm finds corresponding points in 3d, s.t. we know which image patches of 2 different images of the same scene showing the same point in 3d. 
>%%TAGS%%
>
^4rmfdfvp8ba


>%%
>```annotation-json
>{"created":"2023-10-10T11:32:50.651Z","updated":"2023-10-10T11:32:50.651Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":4201,"end":4367},{"type":"TextQuoteSelector","exact":" At test time, we decouple the Detector, whichruns over the whole image in scale space, from the Orientation Estimator andDescriptor, which process only the keypoints","prefix":"ting through the entire network.","suffix":".In the next section we briefly "}]}]}
>```
>%%
>*%%PREFIX%%ting through the entire network.%%HIGHLIGHT%% ==At test time, we decouple the Detector, whichruns over the whole image in scale space, from the Orientation Estimator andDescriptor, which process only the keypoints== %%POSTFIX%%.In the next section we briefly*
>%%LINK%%[[#^ghons5avkcp|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^ghons5avkcp


>%%
>```annotation-json
>{"created":"2023-10-10T16:02:00.797Z","text":"Choose patch size wrt the scale of the keypoint (found by SIFT or similar).","updated":"2023-10-10T16:02:00.797Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":14685,"end":15119},{"type":"TextQuoteSelector","exact":"We extract grayscale training patches according to the scale σ of the point,for both feature and non-feature point image regions. Patches P are extractedfrom a 24σ×24σ support region at these locations, and standardized into S ×Spixels where S = 128. The smaller patches p and pθ that serve as input to theOrientation Estimator and the Descriptor, are cropped and rotated versions ofthese patches, each having size s×s, where s = 64. ","prefix":"those that were not used by SfM.","suffix":"The smaller patches effectivelyc"}]}]}
>```
>%%
>*%%PREFIX%%those that were not used by SfM.%%HIGHLIGHT%% ==We extract grayscale training patches according to the scale σ of the point,for both feature and non-feature point image regions. Patches P are extractedfrom a 24σ×24σ support region at these locations, and standardized into S ×Spixels where S = 128. The smaller patches p and pθ that serve as input to theOrientation Estimator and the Descriptor, are cropped and rotated versions ofthese patches, each having size s×s, where s = 64.== %%POSTFIX%%The smaller patches effectivelyc*
>%%LINK%%[[#^na0pfoeez7|show annotation]]
>%%COMMENT%%
>Choose patch size wrt the scale of the keypoint (found by SIFT or similar).
>%%TAGS%%
>
^na0pfoeez7


>%%
>```annotation-json
>{"created":"2023-10-10T16:15:36.255Z","text":"Use the already trained descriptor for train the orientation estimator using only positive pairs, ie image patches showing the same physical 3d point. The orientation estimator is trained by minimizing the euclidean distance of the embedding-vectors (embedding vectors = output of descriptor)","updated":"2023-10-10T16:15:36.255Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":18753,"end":19501},{"type":"TextQuoteSelector","exact":"and as the Detector is still not trained, we use the image locations from SfM.More formally, we minimize the loss for pairs of corresponding patches, definedas the Euclidean distance between their description vectorsLorientation(P1,x1,P2,x2) = ∥∥hρ(G(P1,x1)) −hρ(G(P2,x2))∥∥2 , (4)where G(P,x) is the patch centered on x after orientation correction: G(P,x) =Rot(P,x,gφ(Crop(P,x))). This complex notation is necessary to properly han-dle the cropping of the image patches. Recall that pairs (P1,P2) comprise imagepatches containing the projections of the same 3D point, and locations x1 andx2 denote the reprojections of these 3D points. As in [9], we do not use pairsthat correspond to different physical points whose orientations are not related.","prefix":"i, E. Trulls, V. Lepetit, P. Fua","suffix":"3.5 DetectorThe Detector takes a"}]}]}
>```
>%%
>*%%PREFIX%%i, E. Trulls, V. Lepetit, P. Fua%%HIGHLIGHT%% ==and as the Detector is still not trained, we use the image locations from SfM.More formally, we minimize the loss for pairs of corresponding patches, definedas the Euclidean distance between their description vectorsLorientation(P1,x1,P2,x2) = ∥∥hρ(G(P1,x1)) −hρ(G(P2,x2))∥∥2 , (4)where G(P,x) is the patch centered on x after orientation correction: G(P,x) =Rot(P,x,gφ(Crop(P,x))). This complex notation is necessary to properly han-dle the cropping of the image patches. Recall that pairs (P1,P2) comprise imagepatches containing the projections of the same 3D point, and locations x1 andx2 denote the reprojections of these 3D points. As in [9], we do not use pairsthat correspond to different physical points whose orientations are not related.== %%POSTFIX%%3.5 DetectorThe Detector takes a*
>%%LINK%%[[#^2kqxs5u0oqf|show annotation]]
>%%COMMENT%%
>Use the already trained descriptor for train the orientation estimator using only positive pairs, ie image patches showing the same physical 3d point. The orientation estimator is trained by minimizing the euclidean distance of the embedding-vectors (embedding vectors = output of descriptor)
>%%TAGS%%
>
^2kqxs5u0oqf


>%%
>```annotation-json
>{"created":"2023-10-10T16:31:25.166Z","updated":"2023-10-10T16:31:25.166Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":21228,"end":21358},{"type":"TextQuoteSelector","exact":"As the Orientation Estimator and the Descriptor have been learned by thispoint, we can train the Detector given the full pipeline.","prefix":"ed Invariant Feature Transform 9","suffix":" To optimize over theparameters "}]}]}
>```
>%%
>*%%PREFIX%%ed Invariant Feature Transform 9%%HIGHLIGHT%% ==As the Orientation Estimator and the Descriptor have been learned by thispoint, we can train the Detector given the full pipeline.== %%POSTFIX%%To optimize over theparameters*
>%%LINK%%[[#^g3fi6sx6ic|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^g3fi6sx6ic


>%%
>```annotation-json
>{"created":"2023-10-10T16:37:46.933Z","text":"L_class = Weird way to classify images: Classify here if keypoint in patch or not. \nL_pair = Minimize distance of corresponding patches ","updated":"2023-10-10T16:37:46.933Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":21853,"end":22306},{"type":"TextQuoteSelector","exact":"detector(P1,P2,P3,P4) = γLclass(P1,P2,P3,P4) + Lpair(P1,P2) , (8)where γ is a hyper-parameter balancing the two terms in this summationLclass(P1,P2,P3,P4) =4∑i=1αi max (0,(1 −softmax (fμ(Pi))yi))2 , (9)with yi = −1 and αi = 3/6 if i = 4, and yi = +1 and αi = 1/6 otherwise tobalance the positives and negatives. softmax is the log-mean-exponential softmaxfunction. We writeLpair(P1,P2) = ‖ hρ(G(P1,softargmax(fμ(P1)))) −hρ(G(P2,softargmax(fμ(P2)))) ‖2 .","prefix":" the sum of theirloss functionsL","suffix":" (10)Note that the locations of "}]}]}
>```
>%%
>*%%PREFIX%%the sum of theirloss functionsL%%HIGHLIGHT%% ==detector(P1,P2,P3,P4) = γLclass(P1,P2,P3,P4) + Lpair(P1,P2) , (8)where γ is a hyper-parameter balancing the two terms in this summationLclass(P1,P2,P3,P4) =4∑i=1αi max (0,(1 −softmax (fμ(Pi))yi))2 , (9)with yi = −1 and αi = 3/6 if i = 4, and yi = +1 and αi = 1/6 otherwise tobalance the positives and negatives. softmax is the log-mean-exponential softmaxfunction. We writeLpair(P1,P2) = ‖ hρ(G(P1,softargmax(fμ(P1)))) −hρ(G(P2,softargmax(fμ(P2)))) ‖2 .== %%POSTFIX%%(10)Note that the locations of*
>%%LINK%%[[#^lt0qfxxhpnm|show annotation]]
>%%COMMENT%%
>L_class = Weird way to classify images: Classify here if keypoint in patch or not. 
>L_pair = Minimize distance of corresponding patches 
>%%TAGS%%
>
^lt0qfxxhpnm


>%%
>```annotation-json
>{"created":"2023-10-10T16:41:47.793Z","updated":"2023-10-10T16:41:47.793Z","document":{"title":"LIFT.pdf","link":[{"href":"urn:x-pdf:2eb99b9f79e5006e7f7def94e58b1939"},{"href":"vault:/Sources/Image Matching/LIFT.pdf"}],"documentFingerprint":"2eb99b9f79e5006e7f7def94e58b1939"},"uri":"vault:/Sources/Image Matching/LIFT.pdf","target":[{"source":"vault:/Sources/Image Matching/LIFT.pdf","selector":[{"type":"TextPositionSelector","start":24060,"end":24454},{"type":"TextQuoteSelector","exact":" In practice, this would betoo expensive. Fortunately, as the Orientation Estimator and the Descriptor onlyneed to be run at local maxima, we can simply decouple the detector from therest to apply it to the full image, and replace the softargmax function by NMS,as outlined in red in Fig. 4. We then apply the Orientation Estimator and theDescriptor only to the patches centered on local maxima","prefix":"dow scheme over the whole image.","suffix":".More exactly, we apply the Dete"}]}]}
>```
>%%
>*%%PREFIX%%dow scheme over the whole image.%%HIGHLIGHT%% ==In practice, this would betoo expensive. Fortunately, as the Orientation Estimator and the Descriptor onlyneed to be run at local maxima, we can simply decouple the detector from therest to apply it to the full image, and replace the softargmax function by NMS,as outlined in red in Fig. 4. We then apply the Orientation Estimator and theDescriptor only to the patches centered on local maxima== %%POSTFIX%%.More exactly, we apply the Dete*
>%%LINK%%[[#^32uj5odnuu|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^32uj5odnuu
