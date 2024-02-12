annotation-target::DELF.pdf

>%%
>```annotation-json
>{"created":"2023-10-06T10:37:54.723Z","text":"Neither detect-than-descript nor detect-and-descript?","updated":"2023-10-06T10:37:54.723Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":5324,"end":5677},{"type":"TextQuoteSelector","exact":"In our approach, the attention model is tightly coupled withthe proposed descriptor; it reuses the same CNN architectureand generates feature scores using very little extra computa-tion (in the spirit of recent advances in object detection [30]).This enables the extraction of both local descriptors and key-points via one forward pass over the network.","prefix":" extraction and image retrieval.","suffix":" We show thatour image retrieval"}]}]}
>```
>%%
>*%%PREFIX%%extraction and image retrieval.%%HIGHLIGHT%% ==In our approach, the attention model is tightly coupled withthe proposed descriptor; it reuses the same CNN architectureand generates feature scores using very little extra computa-tion (in the spirit of recent advances in object detection [30]).This enables the extraction of both local descriptors and key-points via one forward pass over the network.== %%POSTFIX%%We show thatour image retrieval*
>%%LINK%%[[#^iojkpsde52q|show annotation]]
>%%COMMENT%%
>Neither detect-than-descript nor detect-and-descript?
>%%TAGS%%
>
^iojkpsde52q


>%%
>```annotation-json
>{"created":"2023-10-06T11:33:49.016Z","text":"If SuperBrain will work patch-wise, this could refer to patches non existing in one of the images","updated":"2023-10-06T11:33:49.016Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":10357,"end":10673},{"type":"TextQuoteSelector","exact":" In par-ticular, since our query images are collected from personalphoto repositories, some of them may not contain any land-marks and should not retrieve any image from the database.We call these query images distractors, which play a criticalrole to evaluate robustness of algorithms to irrelevant andnoisy queries","prefix":"tially out-of-view objects, etc.","suffix":".We use visual features and GPS "}]}]}
>```
>%%
>*%%PREFIX%%tially out-of-view objects, etc.%%HIGHLIGHT%% ==In par-ticular, since our query images are collected from personalphoto repositories, some of them may not contain any land-marks and should not retrieve any image from the database.We call these query images distractors, which play a criticalrole to evaluate robustness of algorithms to irrelevant andnoisy queries== %%POSTFIX%%.We use visual features and GPS*
>%%LINK%%[[#^9w9nx51yxak|show annotation]]
>%%COMMENT%%
>If SuperBrain will work patch-wise, this could refer to patches non existing in one of the images
>%%TAGS%%
>
^9w9nx51yxak


>%%
>```annotation-json
>{"created":"2023-10-06T11:36:13.555Z","updated":"2023-10-06T11:36:13.555Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":10906,"end":11084},{"type":"TextQuoteSelector","exact":"the location of a query image and the center of the clusterassociated with the retrieved image is less than a threshold,we assume that the two images belong to the same landmark.","prefix":"er. If physical distance between","suffix":"Note that ground-truth annotatio"}]}]}
>```
>%%
>*%%PREFIX%%er. If physical distance between%%HIGHLIGHT%% ==the location of a query image and the center of the clusterassociated with the retrieved image is less than a threshold,we assume that the two images belong to the same landmark.== %%POSTFIX%%Note that ground-truth annotatio*
>%%LINK%%[[#^0afpxe4ryj8|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^0afpxe4ryj8


>%%
>```annotation-json
>{"created":"2023-10-06T11:37:35.091Z","text":"They labeled corresponding images by assigning images the same class if they were taken at the same (~25km) GPS coordinate","updated":"2023-10-06T11:37:35.091Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":11327,"end":11957},{"type":"TextQuoteSelector","exact":"Obviously, this approach for ground-truth construc-tion is noisy due to GPS errors. Also, photos can be capturedfrom a large distance for some landmarks (e.g., Eiffel Tower,Golden Gate Bridge), and consequently the photo locationmight be relatively far from the actual landmark location.However, we found very few incorrect annotations with thethreshold of 25km when checking a subset of data manually.Even though there are few minor errors, it is not problem-atic, especially in relative evaluation, because algorithms areunlikely to be confused between landmarks anyway if theirvisual appearances are sufficiently discriminative","prefix":"ple instances in a singleimage. ","suffix":".4. Image Retrieval with DELFOur"}]}]}
>```
>%%
>*%%PREFIX%%ple instances in a singleimage.%%HIGHLIGHT%% ==Obviously, this approach for ground-truth construc-tion is noisy due to GPS errors. Also, photos can be capturedfrom a large distance for some landmarks (e.g., Eiffel Tower,Golden Gate Bridge), and consequently the photo locationmight be relatively far from the actual landmark location.However, we found very few incorrect annotations with thethreshold of 25km when checking a subset of data manually.Even though there are few minor errors, it is not problem-atic, especially in relative evaluation, because algorithms areunlikely to be confused between landmarks anyway if theirvisual appearances are sufficiently discriminative== %%POSTFIX%%.4. Image Retrieval with DELFOur*
>%%LINK%%[[#^fj08uuoxkad|show annotation]]
>%%COMMENT%%
>They labeled corresponding images by assigning images the same class if they were taken at the same (~25km) GPS coordinate
>%%TAGS%%
>
^fj08uuoxkad


>%%
>```annotation-json
>{"created":"2023-10-06T11:41:57.520Z","text":"Train pretrained CNN ResNet50 with classification task based on the ground-truth coming from the GPS data","updated":"2023-10-06T11:41:57.520Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":12352,"end":13245},{"type":"TextQuoteSelector","exact":"We extract dense features from an image by applying afully convolutional network (FCN), which is constructed byusing the feature extraction layers of a CNN trained witha classification loss. We employ an FCN taken from theResNet50 [13] model, using the output of the conv4 x con-volutional block. To handle scale changes, we explicitlyconstruct an image pyramid and apply the FCN for eachlevel independently. The obtained feature maps are regardedas a dense grid of local descriptors. Features are localizedbased on their receptive fields, which can be computed byconsidering the configuration of convolutional and poolinglayers of the FCN. We use the pixel coordinates of the centerof the receptive field as the feature location. The receptivefield size for the image at the original scale is 291 ×291.Using the image pyramid, we obtain features that describeimage regions of different sizes.","prefix":"nse Localized Feature Extraction","suffix":"We use the original ResNet50 mod"}]}]}
>```
>%%
>*%%PREFIX%%nse Localized Feature Extraction%%HIGHLIGHT%% ==We extract dense features from an image by applying afully convolutional network (FCN), which is constructed byusing the feature extraction layers of a CNN trained witha classification loss. We employ an FCN taken from theResNet50 [13] model, using the output of the conv4 x con-volutional block. To handle scale changes, we explicitlyconstruct an image pyramid and apply the FCN for eachlevel independently. The obtained feature maps are regardedas a dense grid of local descriptors. Features are localizedbased on their receptive fields, which can be computed byconsidering the configuration of convolutional and poolinglayers of the FCN. We use the pixel coordinates of the centerof the receptive field as the feature location. The receptivefield size for the image at the original scale is 291 ×291.Using the image pyramid, we obtain features that describeimage regions of different sizes.== %%POSTFIX%%We use the original ResNet50 mod*
>%%LINK%%[[#^maci466l42|show annotation]]
>%%COMMENT%%
>Train pretrained CNN ResNet50 with classification task based on the ground-truth coming from the GPS data
>%%TAGS%%
>
^maci466l42


>%%
>```annotation-json
>{"created":"2023-10-06T11:42:57.396Z","text":"They claim that through the randomcrop, local descriptors implicitly learn representations that are more relevant for the landmark retrieval problem","updated":"2023-10-06T11:42:57.396Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":13614,"end":13758},{"type":"TextQuoteSelector","exact":"The input images are initially center-cropped to produce square images and rescaled to 250×250.Random 224 ×224 crops are then used for training.","prefix":"ion asillustrated in Fig. 4(a). ","suffix":" As aFeaturesAttention ScoresFea"}]}]}
>```
>%%
>*%%PREFIX%%ion asillustrated in Fig. 4(a).%%HIGHLIGHT%% ==The input images are initially center-cropped to produce square images and rescaled to 250×250.Random 224 ×224 crops are then used for training.== %%POSTFIX%%As aFeaturesAttention ScoresFea*
>%%LINK%%[[#^tzi3pnezgbo|show annotation]]
>%%COMMENT%%
>They claim that through the randomcrop, local descriptors implicitly learn representations that are more relevant for the landmark retrieval problem
>%%TAGS%%
>
^tzi3pnezgbo


>%%
>```annotation-json
>{"created":"2023-10-06T11:48:39.938Z","text":"Using attention scores s.t. only a part of the extracted features (from the classification training) used for keypoint selection","updated":"2023-10-06T11:48:39.938Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":14178,"end":14579},{"type":"TextQuoteSelector","exact":"Instead of using densely extracted features directly forimage retrieval, we design a technique to effectively se-lect a subset of the features. Since a substantial part of thedensely extracted features are irrelevant to our recognitiontask and likely to add clutter, distracting the retrieval pro-cess, keypoint selection is important for both accuracy andcomputational efficiency of retrieval systems","prefix":"tention-based Keypoint Selection","suffix":".4.2.1 Learning with Weak Superv"}]}]}
>```
>%%
>*%%PREFIX%%tention-based Keypoint Selection%%HIGHLIGHT%% ==Instead of using densely extracted features directly forimage retrieval, we design a technique to effectively se-lect a subset of the features. Since a substantial part of thedensely extracted features are irrelevant to our recognitiontask and likely to add clutter, distracting the retrieval pro-cess, keypoint selection is important for both accuracy andcomputational efficiency of retrieval systems== %%POSTFIX%%.4.2.1 Learning with Weak Superv*
>%%LINK%%[[#^jrwcoctgpuq|show annotation]]
>%%COMMENT%%
>Using attention scores s.t. only a part of the extracted features (from the classification training) used for keypoint selection
>%%TAGS%%
>
^jrwcoctgpuq


>%%
>```annotation-json
>{"created":"2023-10-06T11:57:48.451Z","text":"There are N different d-dimensional feature-vectors. We sum these N vectors up to one single vector, but in a weighted way (these vectors are weighted by alpha). The resulting sum, a single d-dimensional vector, gets multiplied with W for the final prediction, which is classification again","updated":"2023-10-06T11:57:48.451Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":15173,"end":15573},{"type":"TextQuoteSelector","exact":"More precisely, we formulate the training as follows. De-note by fn ∈Rd,n = 1,...,N the d-dimensional featuresto be learned jointly with the attention model. Our goal isto learn a score function α(fn; θ) for each feature, where θdenotes the parameters of function α(·). The output logit yof the network is generated by a weighted sum of the featurevectors, which is given byy = W(∑nα(fn; θ) ·fn), (1)","prefix":"oftmax-based landmarkclassifier.","suffix":"where W ∈RM×d represents the wei"}]}]}
>```
>%%
>*%%PREFIX%%oftmax-based landmarkclassifier.%%HIGHLIGHT%% ==More precisely, we formulate the training as follows. De-note by fn ∈Rd,n = 1,...,N the d-dimensional featuresto be learned jointly with the attention model. Our goal isto learn a score function α(fn; θ) for each feature, where θdenotes the parameters of function α(·). The output logit yof the network is generated by a weighted sum of the featurevectors, which is given byy = W(∑nα(fn; θ) ·fn), (1)== %%POSTFIX%%where W ∈RM×d represents the wei*
>%%LINK%%[[#^tb9iebba6do|show annotation]]
>%%COMMENT%%
>There are N different d-dimensional feature-vectors. We sum these N vectors up to one single vector, but in a weighted way (these vectors are weighted by alpha). The resulting sum, a single d-dimensional vector, gets multiplied with W for the final prediction, which is classification again
>%%TAGS%%
>
^tb9iebba6do


>%%
>```annotation-json
>{"created":"2023-10-06T12:10:00.708Z","updated":"2023-10-06T12:10:00.708Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":16709,"end":16867},{"type":"TextQuoteSelector","exact":"While the feature representation and the score func-tion can be trained jointly by backpropagation, we foundthat this setup generates weak models in practice.","prefix":"llenges to the learningprocess. ","suffix":" Therefore,we employ a two-step "}]}]}
>```
>%%
>*%%PREFIX%%llenges to the learningprocess.%%HIGHLIGHT%% ==While the feature representation and the score func-tion can be trained jointly by backpropagation, we foundthat this setup generates weak models in practice.== %%POSTFIX%%Therefore,we employ a two-step*
>%%LINK%%[[#^qphqjv9bika|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^qphqjv9bika


>%%
>```annotation-json
>{"created":"2023-10-06T12:10:16.323Z","text":"Train first descriptors and second the score function with fixed descriptors","updated":"2023-10-06T12:10:16.323Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":16868,"end":17056},{"type":"TextQuoteSelector","exact":"Therefore,we employ a two-step training strategy. First, we learn de-scriptors with fine-tuning as described in Sec. 4.1, and thenthe score function is learned given the fixed descriptors.","prefix":"erates weak models in practice. ","suffix":"Another improvement to our model"}]}]}
>```
>%%
>*%%PREFIX%%erates weak models in practice.%%HIGHLIGHT%% ==Therefore,we employ a two-step training strategy. First, we learn de-scriptors with fine-tuning as described in Sec. 4.1, and thenthe score function is learned given the fixed descriptors.== %%POSTFIX%%Another improvement to our model*
>%%LINK%%[[#^ugking3s9jf|show annotation]]
>%%COMMENT%%
>Train first descriptors and second the score function with fixed descriptors
>%%TAGS%%
>
^ugking3s9jf


>%%
>```annotation-json
>{"created":"2023-10-06T12:11:27.377Z","updated":"2023-10-06T12:11:27.377Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":17056,"end":17482},{"type":"TextQuoteSelector","exact":"Another improvement to our models is obtained by ran-dom image rescaling during attention training process. Thisis intuitive, as the attention model should be able to gener-ate effective scores for features at different scales. In thiscase, the input images are initially center-cropped to pro-duce square images, and rescaled to 900 ×900. Random720 ×720 crops are then extracted and finally randomlyscaled with a factor γ ≤1.","prefix":"ned given the fixed descriptors.","suffix":"4.2.3 CharacteristicsOne unconve"}]}]}
>```
>%%
>*%%PREFIX%%ned given the fixed descriptors.%%HIGHLIGHT%% ==Another improvement to our models is obtained by ran-dom image rescaling during attention training process. Thisis intuitive, as the attention model should be able to gener-ate effective scores for features at different scales. In thiscase, the input images are initially center-cropped to pro-duce square images, and rescaled to 900 ×900. Random720 ×720 crops are then extracted and finally randomlyscaled with a factor γ ≤1.== %%POSTFIX%%4.2.3 CharacteristicsOne unconve*
>%%LINK%%[[#^5tj1wgzpxj5|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^5tj1wgzpxj5


>%%
>```annotation-json
>{"created":"2023-10-06T12:11:56.911Z","updated":"2023-10-06T12:11:56.911Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":17503,"end":17737},{"type":"TextQuoteSelector","exact":"One unconventional aspect of our system is that keypointselection comes after descriptor extraction, which is differentfrom the existing techniques (e.g., SIFT [22] and LIFT [40]),where keypoints are first detected and later described","prefix":"actor γ ≤1.4.2.3 Characteristics","suffix":". Tradi-tional keypoint detector"}]}]}
>```
>%%
>*%%PREFIX%%actor γ ≤1.4.2.3 Characteristics%%HIGHLIGHT%% ==One unconventional aspect of our system is that keypointselection comes after descriptor extraction, which is differentfrom the existing techniques (e.g., SIFT [22] and LIFT [40]),where keypoints are first detected and later described== %%POSTFIX%%. Tradi-tional keypoint detector*
>%%LINK%%[[#^9hypjgnjxjb|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^9hypjgnjxjb


>%%
>```annotation-json
>{"created":"2023-10-06T12:14:51.053Z","text":"They used Nearest neighbor search for finding corresponding images","updated":"2023-10-06T12:14:51.053Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":19115,"end":19176},{"type":"TextQuoteSelector","exact":"Our imageretrieval system is based on nearest neighbor search","prefix":" scores per image are selected. ","suffix":", whichis implemented by a combi"}]}]}
>```
>%%
>*%%PREFIX%%scores per image are selected.%%HIGHLIGHT%% ==Our imageretrieval system is based on nearest neighbor search== %%POSTFIX%%, whichis implemented by a combi*
>%%LINK%%[[#^e78cl8ao4bn|show annotation]]
>%%COMMENT%%
>They used Nearest neighbor search for finding corresponding images
>%%TAGS%%
>
^e78cl8ao4bn


>%%
>```annotation-json
>{"created":"2023-10-06T12:17:40.992Z","updated":"2023-10-06T12:17:40.992Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":20470,"end":20818},{"type":"TextQuoteSelector","exact":"This pipeline requires less than 8GB memory to index 1billion descriptors, which is sufficient to handle our large-scale landmark dataset. The latency of the nearest neighborsearch is less than 2 seconds using a single CPU under ourexperiment setup, where we soft-assign 5 centroids to eachquery and search up to 10K leaf nodes within each inverted","prefix":"th theones from landmark images.","suffix":"index tree.5. ExperimentsThis se"}]}]}
>```
>%%
>*%%PREFIX%%th theones from landmark images.%%HIGHLIGHT%% ==This pipeline requires less than 8GB memory to index 1billion descriptors, which is sufficient to handle our large-scale landmark dataset. The latency of the nearest neighborsearch is less than 2 seconds using a single CPU under ourexperiment setup, where we soft-assign 5 centroids to eachquery and search up to 10K leaf nodes within each inverted== %%POSTFIX%%index tree.5. ExperimentsThis se*
>%%LINK%%[[#^7svf5m70qzt|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^7svf5m70qzt


>%%
>```annotation-json
>{"created":"2023-10-06T12:37:48.321Z","updated":"2023-10-06T12:37:48.321Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":26842,"end":27115},{"type":"TextQuoteSelector","exact":"In particular, note that the useof attention is more important than fine-tuning. This demon-strates that the proposed attention layers effectively learn toselect the most discriminative features for the retrieval task,even if the features are simply pretrained on ImageNet.","prefix":"ions toperformance improvement. ","suffix":"In terms of memory requirement, "}]}]}
>```
>%%
>*%%PREFIX%%ions toperformance improvement.%%HIGHLIGHT%% ==In particular, note that the useof attention is more important than fine-tuning. This demon-strates that the proposed attention layers effectively learn toselect the most discriminative features for the retrieval task,even if the features are simply pretrained on ImageNet.== %%POSTFIX%%In terms of memory requirement,*
>%%LINK%%[[#^1dy8lucj9vk|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^1dy8lucj9vk


>%%
>```annotation-json
>{"created":"2023-10-06T12:38:40.419Z","updated":"2023-10-06T12:38:40.419Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":31247,"end":31314},{"type":"TextQuoteSelector","exact":"Both ends of the red lines denote the centers of matching features.","prefix":"bjects, and background clutter. ","suffix":" Since the receptivefields are f"}]}]}
>```
>%%
>*%%PREFIX%%bjects, and background clutter.%%HIGHLIGHT%% ==Both ends of the red lines denote the centers of matching features.== %%POSTFIX%%Since the receptivefields are f*
>%%LINK%%[[#^qj30dhamuw|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^qj30dhamuw


>%%
>```annotation-json
>{"created":"2023-10-06T13:14:24.099Z","updated":"2023-10-06T13:14:24.099Z","document":{"title":"DELF.pdf","link":[{"href":"urn:x-pdf:93a296266cc4a59b8eeda2d08e356ffe"},{"href":"vault:/Sources/Image Matching/DELF.pdf"}],"documentFingerprint":"93a296266cc4a59b8eeda2d08e356ffe"},"uri":"vault:/Sources/Image Matching/DELF.pdf","target":[{"source":"vault:/Sources/Image Matching/DELF.pdf","selector":[{"type":"TextPositionSelector","start":4974,"end":5165},{"type":"TextQuoteSelector","exact":"We then propose a CNN-based local feature with atten-tion, which is trained with weak supervision using image-level class labels only, without the need of object- and patch-level annotations.","prefix":"ot necessarily depict landmarks.","suffix":" This new feature descriptor is "}]}]}
>```
>%%
>*%%PREFIX%%ot necessarily depict landmarks.%%HIGHLIGHT%% ==We then propose a CNN-based local feature with atten-tion, which is trained with weak supervision using image-level class labels only, without the need of object- and patch-level annotations.== %%POSTFIX%%This new feature descriptor is*
>%%LINK%%[[#^zmjmxqwaci|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^zmjmxqwaci
