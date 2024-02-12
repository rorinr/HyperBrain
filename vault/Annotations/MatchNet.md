annotation-target::MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf

>%%
>```annotation-json
>{"created":"2023-09-22T09:51:55.413Z","text":"Instead of directly computing the l2 norm between descriptors, the descriptors are feed again in a FCN (which is the here called learned distance metric)\n","updated":"2023-09-22T09:51:55.413Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":3521,"end":3905},{"type":"TextQuoteSelector","exact":"In our system,dubbed MatchNet, each patch passes through a convolu-tional network to generate a fixed-dimensional representa-tion reminiscent of SIFT. However, unlike in SIFT, wheretwo descriptors are compared in feature space using the Eu-clidean distance, in MatchNet, the representations are com-pared using a learned distance metric, implemented as a setof fully connected layers.","prefix":" for robust feature comparison. ","suffix":"Our contributions include: 1) A "}]}]}
>```
>%%
>*%%PREFIX%%for robust feature comparison.%%HIGHLIGHT%% ==In our system,dubbed MatchNet, each patch passes through a convolu-tional network to generate a fixed-dimensional representa-tion reminiscent of SIFT. However, unlike in SIFT, wheretwo descriptors are compared in feature space using the Eu-clidean distance, in MatchNet, the representations are com-pared using a learned distance metric, implemented as a setof fully connected layers.== %%POSTFIX%%Our contributions include: 1) A*
>%%LINK%%[[#^g0fanxbw6bu|show annotation]]
>%%COMMENT%%
>Instead of directly computing the l2 norm between descriptors, the descriptors are feed again in a FCN (which is the here called learned distance metric)
>
>%%TAGS%%
>
^g0fanxbw6bu


>%%
>```annotation-json
>{"created":"2023-09-22T10:03:14.550Z","updated":"2023-09-22T10:03:14.550Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":13929,"end":14098},{"type":"TextQuoteSelector","exact":"The preprocessing layer: Following a previous conven-tion, for each pixel in the input grayscale patch we normal-ize its intensity value x (in [0, 255]) to (x −128)/160.","prefix":" 5 and plot results in Figure 4.","suffix":"4. Training and predictionThe fe"}]}]}
>```
>%%
>*%%PREFIX%%5 and plot results in Figure 4.%%HIGHLIGHT%% ==The preprocessing layer: Following a previous conven-tion, for each pixel in the input grayscale patch we normal-ize its intensity value x (in [0, 255]) to (x −128)/160.== %%POSTFIX%%4. Training and predictionThe fe*
>%%LINK%%[[#^ewyy62j4lj|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^ewyy62j4lj


>%%
>```annotation-json
>{"created":"2023-09-22T10:05:53.331Z","text":"There are way more negative examples at test time (when using bruteforce approach). ","updated":"2023-09-22T10:05:53.331Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":15650,"end":15913},{"type":"TextQuoteSelector","exact":"Sampling is important in training, as the matching (+)and non-matching (-) pairs are highly unbalanced. We use asampler to generate equal number of positives and negativesin each mini-batch so that the network will not be overly bi-ased towards negative decisions","prefix":"etwork.4.1. Sampling in training","suffix":". The sampler also enforcesvarie"}]}]}
>```
>%%
>*%%PREFIX%%etwork.4.1. Sampling in training%%HIGHLIGHT%% ==Sampling is important in training, as the matching (+)and non-matching (-) pairs are highly unbalanced. We use asampler to generate equal number of positives and negativesin each mini-batch so that the network will not be overly bi-ased towards negative decisions== %%POSTFIX%%. The sampler also enforcesvarie*
>%%LINK%%[[#^2h3xrecfltq|show annotation]]
>%%COMMENT%%
>There are way more negative examples at test time (when using bruteforce approach). 
>%%TAGS%%
>
^2h3xrecfltq


>%%
>```annotation-json
>{"created":"2023-09-22T10:10:10.219Z","updated":"2023-09-22T10:10:10.219Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":17279,"end":17855},{"type":"TextQuoteSelector","exact":"For instance, if the batch size is 32, in each training it-eration we feed SGD 16 positives and 16 negatives. Thepositives are obtained by reading the next 16 groups fromthe database and randomly picking one pair in each group.Since we go through the whole dataset many times, eventhough we only pick one positive pair from each group ineach pass, the network still gets good positive coverage,especially when the average group size is small. The 16negatives are obtained by sampling two pairs from differentgroups from the reservoir buffer that stores previous loadedpatches.","prefix":"edureis detailed in Algorithm 1.","suffix":" At the first few iterations, th"}]}]}
>```
>%%
>*%%PREFIX%%edureis detailed in Algorithm 1.%%HIGHLIGHT%% ==For instance, if the batch size is 32, in each training it-eration we feed SGD 16 positives and 16 negatives. Thepositives are obtained by reading the next 16 groups fromthe database and randomly picking one pair in each group.Since we go through the whole dataset many times, eventhough we only pick one positive pair from each group ineach pass, the network still gets good positive coverage,especially when the average group size is small. The 16negatives are obtained by sampling two pairs from differentgroups from the reservoir buffer that stores previous loadedpatches.== %%POSTFIX%%At the first few iterations, th*
>%%LINK%%[[#^uslinhtf1nn|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^uslinhtf1nn


>%%
>```annotation-json
>{"created":"2023-09-22T10:24:41.236Z","updated":"2023-09-22T10:24:41.236Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":28895,"end":29007},{"type":"TextQuoteSelector","exact":"The 4096-512x512 model should be used if thefeature dimension is not a concern, or if accuracy takes pri-ority. ","prefix":"storage/computationconstraints: ","suffix":"This model outperforms others by"}]}]}
>```
>%%
>*%%PREFIX%%storage/computationconstraints:%%HIGHLIGHT%% ==The 4096-512x512 model should be used if thefeature dimension is not a concern, or if accuracy takes pri-ority.== %%POSTFIX%%This model outperforms others by*
>%%LINK%%[[#^tb30e24du8j|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^tb30e24du8j


>%%
>```annotation-json
>{"created":"2023-09-22T10:24:57.846Z","text":"Use SIFT and variations of sift as baselines to compare the nn to","updated":"2023-09-22T10:24:57.846Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":21162,"end":21290},{"type":"TextQuoteSelector","exact":"SIFT baselines. We use VLFeat [31]’s vlsift() withdefault parameters and custom frame input to extract SIFTdescriptor on patches","prefix":"rror@95%), the lower the better.","suffix":". The frame center is the center"}]}]}
>```
>%%
>*%%PREFIX%%rror@95%), the lower the better.%%HIGHLIGHT%% ==SIFT baselines. We use VLFeat [31]’s vlsift() withdefault parameters and custom frame input to extract SIFTdescriptor on patches== %%POSTFIX%%. The frame center is the center*
>%%LINK%%[[#^wm71r3fmf8g|show annotation]]
>%%COMMENT%%
>Use SIFT and variations of sift as baselines to compare the nn to
>%%TAGS%%
>
^wm71r3fmf8g


>%%
>```annotation-json
>{"created":"2023-09-22T10:28:34.863Z","text":"They quantized the embeddings to reduce the required memory","updated":"2023-09-22T10:28:34.863Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/Image Matching/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":22746,"end":23523},{"type":"TextQuoteSelector","exact":"MatchNet with quantized features. We evaluate theperformance of MatchNet with quantized features. The out-put features of the bottleneck layer in the feature tower (Fig-ure 1-A) are represented as floating point numbers. Theyare the outputs of ReLu units, thus the values are alwaysnon-negative. We quantize these feature values in a simplis-tic way. For a trained network, we compute the maximumvalue M for the features across all dimensions on a set ofrandom patches in the training set. Then each element vin the feature is quantized as q(v) = min(2n −1, ⌊(2n −1)v/M ⌋), where n is the number of bits we quantize thefeature to. When the feature is fed to the metric network, vis restored using q(v)M/(2n −1). We evaluate the perfor-mance using different quantization levels.","prefix":"rk without the bottleneck layer.","suffix":"The quantized features give us a"}]}]}
>```
>%%
>*%%PREFIX%%rk without the bottleneck layer.%%HIGHLIGHT%% ==MatchNet with quantized features. We evaluate theperformance of MatchNet with quantized features. The out-put features of the bottleneck layer in the feature tower (Fig-ure 1-A) are represented as floating point numbers. Theyare the outputs of ReLu units, thus the values are alwaysnon-negative. We quantize these feature values in a simplis-tic way. For a trained network, we compute the maximumvalue M for the features across all dimensions on a set ofrandom patches in the training set. Then each element vin the feature is quantized as q(v) = min(2n −1, ⌊(2n −1)v/M ⌋), where n is the number of bits we quantize thefeature to. When the feature is fed to the metric network, vis restored using q(v)M/(2n −1). We evaluate the perfor-mance using different quantization levels.== %%POSTFIX%%The quantized features give us a*
>%%LINK%%[[#^gecfoylti35|show annotation]]
>%%COMMENT%%
>They quantized the embeddings to reduce the required memory
>%%TAGS%%
>
^gecfoylti35


>%%
>```annotation-json
>{"created":"2023-10-03T06:31:57.332Z","text":"Binary classification for metric network","updated":"2023-10-03T06:31:57.332Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":11319,"end":11512},{"type":"TextQuoteSelector","exact":"0, 1] from the two units of FC3, These are non-negative,sum up to one, and can be interpreted as the network’s es-timate of probability that the two patches match and do notmatch, respectively.","prefix":"tagepipeline (See Section 4.2).[","suffix":"Two-tower structure with tied pa"}]}]}
>```
>%%
>*%%PREFIX%%tagepipeline (See Section 4.2).[%%HIGHLIGHT%% ==0, 1] from the two units of FC3, These are non-negative,sum up to one, and can be interpreted as the network’s es-timate of probability that the two patches match and do notmatch, respectively.== %%POSTFIX%%Two-tower structure with tied pa*
>%%LINK%%[[#^g8uwloubndd|show annotation]]
>%%COMMENT%%
>Binary classification for metric network
>%%TAGS%%
>
^g8uwloubndd


>%%
>```annotation-json
>{"created":"2023-10-03T07:50:35.742Z","updated":"2023-10-03T07:50:35.742Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":16287,"end":16568},{"type":"TextQuoteSelector","exact":"for negative sampling, we use a reservoir sam-pler [32] with a buffer size of R patches. At any time T thebuffer maintains R patches as if uniformly sampled from thepatch stream up to T , allowing a variety of non-matchingpairs to be generated efficiently. The buffer size controls","prefix":"andomly pick two fromthe group; ","suffix":"Algorithm 1 Generate a batch of "}]}]}
>```
>%%
>*%%PREFIX%%andomly pick two fromthe group;%%HIGHLIGHT%% ==for negative sampling, we use a reservoir sam-pler [32] with a buffer size of R patches. At any time T thebuffer maintains R patches as if uniformly sampled from thepatch stream up to T , allowing a variety of non-matchingpairs to be generated efficiently. The buffer size controls== %%POSTFIX%%Algorithm 1 Generate a batch of*
>%%LINK%%[[#^twlf1y6wkb|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^twlf1y6wkb


>%%
>```annotation-json
>{"created":"2023-10-03T07:54:35.580Z","updated":"2023-10-03T07:54:35.580Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":19221,"end":19334},{"type":"TextQuoteSelector","exact":"Figure 3. MatchNet is disassembled during prediction. The featurenetwork and the metric network run in a pipeline","prefix":"Bn1 n2n1n2n1 x n2Feature pairs2B","suffix":".stages (Figure 3). First we gen"}]}]}
>```
>%%
>*%%PREFIX%%Bn1 n2n1n2n1 x n2Feature pairs2B%%HIGHLIGHT%% ==Figure 3. MatchNet is disassembled during prediction. The featurenetwork and the metric network run in a pipeline== %%POSTFIX%%.stages (Figure 3). First we gen*
>%%LINK%%[[#^pggfaqe9cj|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^pggfaqe9cj


>%%
>```annotation-json
>{"created":"2023-10-03T07:58:29.743Z","updated":"2023-10-03T07:58:29.743Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":12434,"end":13003},{"type":"TextQuoteSelector","exact":"Table 1. Layer parameters of MatchNet. The output dimensionis given by (height × width × depth). PS: patch size for con-volution and pooling layers; S: stride. Layer types: C: convo-lution, MP: max-pooling, FC: fully-connected. We always padthe convolution and pooling layers so the output height and widthare those of the input divided by the stride. For FC layers,their size B and F are chosen from: B ∈ {64, 128, 256, 512},F ∈ {128, 256, 512, 1024}. All convolution and FC layers useReLU activation except for FC3, whose output is normalized withSoftmax (Equation 2)","prefix":"uction problems, and it also has","suffix":".Name Type Output Dim. PS SConv0"}]}]}
>```
>%%
>*%%PREFIX%%uction problems, and it also has%%HIGHLIGHT%% ==Table 1. Layer parameters of MatchNet. The output dimensionis given by (height × width × depth). PS: patch size for con-volution and pooling layers; S: stride. Layer types: C: convo-lution, MP: max-pooling, FC: fully-connected. We always padthe convolution and pooling layers so the output height and widthare those of the input divided by the stride. For FC layers,their size B and F are chosen from: B ∈ {64, 128, 256, 512},F ∈ {128, 256, 512, 1024}. All convolution and FC layers useReLU activation except for FC3, whose output is normalized withSoftmax (Equation 2)== %%POSTFIX%%.Name Type Output Dim. PS SConv0*
>%%LINK%%[[#^q11wm2o5s8|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^q11wm2o5s8


>%%
>```annotation-json
>{"created":"2023-10-03T08:04:49.468Z","text":"evaluation metric","updated":"2023-10-03T08:04:49.468Z","document":{"title":"MatchNet: Unifying feature and metric learning for patch-based matching","link":[{"href":"urn:x-pdf:c79ee13608e35f198bedd33380167540"},{"href":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf"}],"documentFingerprint":"c79ee13608e35f198bedd33380167540"},"uri":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","target":[{"source":"vault:/Sources/MatchNet_Unifying_feature_and_metric_learning_for_patch-based_matching.pdf","selector":[{"type":"TextPositionSelector","start":21053,"end":21162},{"type":"TextQuoteSelector","exact":"Thecommonly used evaluation metric is the false positive rateat 95% recall (Error@95%), the lower the better.","prefix":"beled pairs in the test subset. ","suffix":"SIFT baselines. We use VLFeat [3"}]}]}
>```
>%%
>*%%PREFIX%%beled pairs in the test subset.%%HIGHLIGHT%% ==Thecommonly used evaluation metric is the false positive rateat 95% recall (Error@95%), the lower the better.== %%POSTFIX%%SIFT baselines. We use VLFeat [3*
>%%LINK%%[[#^txjgsc09fjb|show annotation]]
>%%COMMENT%%
>evaluation metric
>%%TAGS%%
>
^txjgsc09fjb
