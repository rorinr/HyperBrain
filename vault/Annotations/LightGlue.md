annotation-target::LightGlue.pdf

>%%
>```annotation-json
>{"created":"2023-11-08T17:47:33.387Z","updated":"2023-11-08T17:47:33.387Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":1291,"end":1655},{"type":"TextQuoteSelector","exact":"Reliably describingeach point is challenging in conditions that exhibit symme-tries, weak texture, or appearance changes due to varyingviewpoint and lighting. To reject outliers that arise fromocclusion and missing points, such representations shouldalso be discriminative. This yields two conflicting objectives,robustness and uniqueness, that are hard to satisfy","prefix":" their local visual appearance. ","suffix":".To address these limitations, S"}]}]}
>```
>%%
>*%%PREFIX%%their local visual appearance.%%HIGHLIGHT%% ==Reliably describingeach point is challenging in conditions that exhibit symme-tries, weak texture, or appearance changes due to varyingviewpoint and lighting. To reject outliers that arise fromocclusion and missing points, such representations shouldalso be discriminative. This yields two conflicting objectives,robustness and uniqueness, that are hard to satisfy== %%POSTFIX%%.To address these limitations, S*
>%%LINK%%[[#^iuelfmuuji|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^iuelfmuuji


>%%
>```annotation-json
>{"created":"2023-11-08T17:49:25.622Z","updated":"2023-11-08T17:49:25.622Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":1686,"end":1836},{"type":"TextQuoteSelector","exact":"SuperGlue [56] introduced anew paradigm – a deep network that considers both imagesat the same time to jointly match sparse points and rejectoutliers.","prefix":"y.To address these limitations, ","suffix":" It leverages the powerful Trans"}]}]}
>```
>%%
>*%%PREFIX%%y.To address these limitations,%%HIGHLIGHT%% ==SuperGlue [56] introduced anew paradigm – a deep network that considers both imagesat the same time to jointly match sparse points and rejectoutliers.== %%POSTFIX%%It leverages the powerful Trans*
>%%LINK%%[[#^kia80aefbeg|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kia80aefbeg


>%%
>```annotation-json
>{"created":"2023-11-08T17:52:19.029Z","text":"The model could stop after 3 layers or after 8, dependent on how hard a specific image pair is to match. ","updated":"2023-11-08T17:52:19.029Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":3927,"end":4026},{"type":"TextQuoteSelector","exact":"Unlike previous approaches, LightGlue is adaptive to thedifficulty of each image pair, which varies","prefix":"istingsparse and dense matchers.","suffix":" based on the1arXiv:2306.13643v1"}]}]}
>```
>%%
>*%%PREFIX%%istingsparse and dense matchers.%%HIGHLIGHT%% ==Unlike previous approaches, LightGlue is adaptive to thedifficulty of each image pair, which varies== %%POSTFIX%%based on the1arXiv:2306.13643v1*
>%%LINK%%[[#^5ah67mb4z4h|show annotation]]
>%%COMMENT%%
>The model could stop after 3 layers or after 8, dependent on how hard a specific image pair is to match. 
>%%TAGS%%
>
^5ah67mb4z4h


>%%
>```annotation-json
>{"created":"2023-11-08T17:52:56.368Z","updated":"2023-11-08T17:52:56.368Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":4543,"end":4615},{"type":"TextQuoteSelector","exact":" a behavior that is reminiscent of howhumans process visual information.","prefix":"o match thanon challenging ones,","suffix":" This is achieved by 1)predictin"}]}]}
>```
>%%
>*%%PREFIX%%o match thanon challenging ones,%%HIGHLIGHT%% ==a behavior that is reminiscent of howhumans process visual information.== %%POSTFIX%%This is achieved by 1)predictin*
>%%LINK%%[[#^i8lxhaq1kbd|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^i8lxhaq1kbd


>%%
>```annotation-json
>{"created":"2023-11-08T17:56:37.554Z","text":"LightGlue is able to stop working on on image-set when enough matches are found. It does so by predicting correspondences after each block. It also introspects these predictions and predict wheter further computation is required. LightGlue is able to discard not matchable point and focus then on covisible areas.","updated":"2023-11-08T17:56:37.554Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":4616,"end":4924},{"type":"TextQuoteSelector","exact":"This is achieved by 1)predicting a set of correspondences after each computationalblocks, and 2) enabling the model to introspect them andpredict whether further computation is required. LigthGluealso discards at an early stage points that are not matchable,thus focusing its attention on the covisible area.","prefix":"ans process visual information. ","suffix":"Our experiments show that LightG"}]}]}
>```
>%%
>*%%PREFIX%%ans process visual information.%%HIGHLIGHT%% ==This is achieved by 1)predicting a set of correspondences after each computationalblocks, and 2) enabling the model to introspect them andpredict whether further computation is required. LigthGluealso discards at an early stage points that are not matchable,thus focusing its attention on the covisible area.== %%POSTFIX%%Our experiments show that LightG*
>%%LINK%%[[#^2unsl0hphjt|show annotation]]
>%%COMMENT%%
>LightGlue is able to stop working on on image-set when enough matches are found. It does so by predicting correspondences after each block. It also introspects these predictions and predict wheter further computation is required. LightGlue is able to discard not matchable point and focus then on covisible areas.
>%%TAGS%%
>
^2unsl0hphjt


>%%
>```annotation-json
>{"created":"2023-11-09T08:17:07.001Z","updated":"2023-11-09T08:17:07.001Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":6247,"end":6444},{"type":"TextQuoteSelector","exact":"ome correspondences areincorrect. Those are filtered out by heuristics, like Lowe’sratio test [41] or the mutual check, inlier classifiers [44, 82],and by robustly fitting geometric models [22, 7].","prefix":"nts and imperfect descriptors, s","suffix":" This pro-cess requires extensiv"}]}]}
>```
>%%
>*%%PREFIX%%nts and imperfect descriptors, s%%HIGHLIGHT%% ==ome correspondences areincorrect. Those are filtered out by heuristics, like Lowe’sratio test [41] or the mutual check, inlier classifiers [44, 82],and by robustly fitting geometric models [22, 7].== %%POSTFIX%%This pro-cess requires extensiv*
>%%LINK%%[[#^gj1npnydszb|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^gj1npnydszb


>%%
>```annotation-json
>{"created":"2023-11-09T08:21:38.166Z","updated":"2023-11-09T08:21:38.166Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":7880,"end":8131},{"type":"TextQuoteSelector","exact":"Conversely, dense matchers like LoFTR [68] and follow-ups [9, 78] match points distributed on dense grids ratherthan sparse locations. This boosts the robustness to impres-sive levels but is generally much slower because it processesmany more elements","prefix":"f reducing its overall capacity.","suffix":". This limits the resolution of "}]}]}
>```
>%%
>*%%PREFIX%%f reducing its overall capacity.%%HIGHLIGHT%% ==Conversely, dense matchers like LoFTR [68] and follow-ups [9, 78] match points distributed on dense grids ratherthan sparse locations. This boosts the robustness to impres-sive levels but is generally much slower because it processesmany more elements== %%POSTFIX%%. This limits the resolution of*
>%%LINK%%[[#^do8mk08c0gg|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^do8mk08c0gg


>%%
>```annotation-json
>{"created":"2023-11-09T08:21:58.699Z","updated":"2023-11-09T08:21:58.699Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":8289,"end":8393},{"type":"TextQuoteSelector","exact":"that fair tuning and evaluation makes it competitive withdense matchers, for a fraction of the run time.","prefix":"erates on sparse inputs, we show","suffix":"Making Transformers efficient ha"}]}]}
>```
>%%
>*%%PREFIX%%erates on sparse inputs, we show%%HIGHLIGHT%% ==that fair tuning and evaluation makes it competitive withdense matchers, for a fraction of the run time.== %%POSTFIX%%Making Transformers efficient ha*
>%%LINK%%[[#^wk8vhuwm9b|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^wk8vhuwm9b


>%%
>```annotation-json
>{"created":"2023-11-09T08:23:52.690Z","updated":"2023-11-09T08:23:52.690Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":13410,"end":13617},{"type":"TextQuoteSelector","exact":"Whileabsolute sinusoidal [74] or learned encodings [17, 51] wereinitially prevalent, recent works have studied relative en-codings [63, 67] to stabilize the training and better capturelong-range dependencies","prefix":"a large impact on the accuracy. ","suffix":".LightGlue adapts some of these "}]}]}
>```
>%%
>*%%PREFIX%%a large impact on the accuracy.%%HIGHLIGHT%% ==Whileabsolute sinusoidal [74] or learned encodings [17, 51] wereinitially prevalent, recent works have studied relative en-codings [63, 67] to stabilize the training and better capturelong-range dependencies== %%POSTFIX%%.LightGlue adapts some of these*
>%%LINK%%[[#^vl591mxwt6g|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^vl591mxwt6g


>%%
>```annotation-json
>{"created":"2023-11-09T08:29:35.964Z","updated":"2023-11-09T08:29:35.964Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":13776,"end":13904},{"type":"TextQuoteSelector","exact":"LightGlue predicts a partial assign-ment between two sets of local features extracted from im-ages A and B, following SuperGlue.","prefix":"re matchingProblem formulation: ","suffix":" Each local feature i iscomposed"}]}]}
>```
>%%
>*%%PREFIX%%re matchingProblem formulation:%%HIGHLIGHT%% ==LightGlue predicts a partial assign-ment between two sets of local features extracted from im-ages A and B, following SuperGlue.== %%POSTFIX%%Each local feature i iscomposed*
>%%LINK%%[[#^x5ttf4do4t8|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^x5ttf4do4t8


>%%
>```annotation-json
>{"created":"2023-11-09T08:31:22.325Z","updated":"2023-11-09T08:31:22.325Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":14541,"end":14929},{"type":"TextQuoteSelector","exact":"LightGlue is made of a stack of Lidentical layers that process the two sets jointly. Each layeris composed of self- and cross-attention units that updatethe representation of each point. A classifier then decides,at each layer, whether to halt the inference, thus avoidingunnecessary computations. A lightweight head finally com-putes a partial assignment from the set of representations.","prefix":"spondences.Overview – Figure 3: ","suffix":"3.1. Transformer backboneWe asso"}]}]}
>```
>%%
>*%%PREFIX%%spondences.Overview – Figure 3:%%HIGHLIGHT%% ==LightGlue is made of a stack of Lidentical layers that process the two sets jointly. Each layeris composed of self- and cross-attention units that updatethe representation of each point. A classifier then decides,at each layer, whether to halt the inference, thus avoidingunnecessary computations. A lightweight head finally com-putes a partial assignment from the set of representations.== %%POSTFIX%%3.1. Transformer backboneWe asso*
>%%LINK%%[[#^jxxllnz0ha|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^jxxllnz0ha


>%%
>```annotation-json
>{"created":"2023-11-09T08:37:31.538Z","updated":"2023-11-09T08:37:31.538Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":18620,"end":18794},{"type":"TextQuoteSelector","exact":"This score encodes the likelihood of i to have a correspond-ing point. A point that is not detected in the other image,e.g. when occluded, is not matchable and thus has σi →0","prefix":"igmoid (Linear(xi)) ∈[0,1] . (7)","suffix":".Correspondences: We combine bot"}]}]}
>```
>%%
>*%%PREFIX%%igmoid (Linear(xi)) ∈[0,1] . (7)%%HIGHLIGHT%% ==This score encodes the likelihood of i to have a correspond-ing point. A point that is not detected in the other image,e.g. when occluded, is not matchable and thus has σi →0== %%POSTFIX%%.Correspondences: We combine bot*
>%%LINK%%[[#^4s3m7xna3ct|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^4s3m7xna3ct


>%%
>```annotation-json
>{"created":"2023-11-09T08:41:19.828Z","updated":"2023-11-09T08:41:19.828Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":21042,"end":21120},{"type":"TextQuoteSelector","exact":"threshold α directly controls the trade-off between accuracyand inference time","prefix":"cy of each classifier. The exit4","suffix":".Point pruning: When the exit cr"}]}]}
>```
>%%
>*%%PREFIX%%cy of each classifier. The exit4%%HIGHLIGHT%% ==threshold α directly controls the trade-off between accuracyand inference time== %%POSTFIX%%.Point pruning: When the exit cr*
>%%LINK%%[[#^d5brgoxavkc|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^d5brgoxavkc


>%%
>```annotation-json
>{"created":"2023-11-09T08:42:22.010Z","updated":"2023-11-09T08:42:22.010Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":21609,"end":21834},{"type":"TextQuoteSelector","exact":"We train LightGlue in two stages: we first train it to pre-dict correspondences and only after train the confidenceclassifier. The latter thus does not impact the accuracy at thefinal layer or the convergence of the training.","prefix":"ct the accuracy.3.4. Supervision","suffix":"Correspondences: We supervise th"}]}]}
>```
>%%
>*%%PREFIX%%ct the accuracy.3.4. Supervision%%HIGHLIGHT%% ==We train LightGlue in two stages: we first train it to pre-dict correspondences and only after train the confidenceclassifier. The latter thus does not impact the accuracy at thefinal layer or the convergence of the training.== %%POSTFIX%%Correspondences: We supervise th*
>%%LINK%%[[#^8bvbzgmpzwp|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^8bvbzgmpzwp


>%%
>```annotation-json
>{"created":"2023-11-09T08:44:04.000Z","updated":"2023-11-09T08:44:04.000Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":22055,"end":22166},{"type":"TextQuoteSelector","exact":"Ground truth matches Mare pairs of points with a low re-projection error in both images and a consistent depth.","prefix":"ints from A to B and conversely.","suffix":" Somepoints  ̄A⊆Aand  ̄B ⊆Bare l"}]}]}
>```
>%%
>*%%PREFIX%%ints from A to B and conversely.%%HIGHLIGHT%% ==Ground truth matches Mare pairs of points with a low re-projection error in both images and a consistent depth.== %%POSTFIX%%Somepoints  ̄A⊆Aand  ̄B ⊆Bare l*
>%%LINK%%[[#^wh94mqoc6fm|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^wh94mqoc6fm


>%%
>```annotation-json
>{"created":"2023-11-09T08:44:58.730Z","updated":"2023-11-09T08:44:58.730Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":24585,"end":24710},{"type":"TextQuoteSelector","exact":"Because of how expensive Sinkhorn is,SuperGlue cannot make predictions after each layer and issupervised only at the last one","prefix":"ner gradients.Deep supervision: ","suffix":". The lighter head of LightGluem"}]}]}
>```
>%%
>*%%PREFIX%%ner gradients.Deep supervision:%%HIGHLIGHT%% ==Because of how expensive Sinkhorn is,SuperGlue cannot make predictions after each layer and issupervised only at the last one== %%POSTFIX%%. The lighter head of LightGluem*
>%%LINK%%[[#^3sg67grkd35|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^3sg67grkd35


>%%
>```annotation-json
>{"created":"2023-11-13T14:32:37.176Z","updated":"2023-11-13T14:32:37.176Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":2450,"end":2470},{"type":"TextQuoteSelector","exact":"Figure 1. LightGlue ","prefix":"-depthadaptiveoptimizedLightGlue","suffix":"matches sparse features faster a"}]}]}
>```
>%%
>*%%PREFIX%%-depthadaptiveoptimizedLightGlue%%HIGHLIGHT%% ==Figure 1. LightGlue== %%POSTFIX%%matches sparse features faster a*
>%%LINK%%[[#^l5q3vruge1|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^l5q3vruge1


>%%
>```annotation-json
>{"created":"2023-11-13T15:06:57.516Z","updated":"2023-11-13T15:06:57.516Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":8910,"end":9085},{"type":"TextQuoteSelector","exact":"Other, orthogonal works instead adaptively modulatethe network depth by predicting whether the prediction ofa token at a given layer is final or requires further com-putations","prefix":"o drastically speeds it up [14].","suffix":" [15, 20, 62] . This is mostly i"}]}]}
>```
>%%
>*%%PREFIX%%o drastically speeds it up [14].%%HIGHLIGHT%% ==Other, orthogonal works instead adaptively modulatethe network depth by predicting whether the prediction ofa token at a given layer is final or requires further com-putations== %%POSTFIX%%[15, 20, 62] . This is mostly i*
>%%LINK%%[[#^a5yqph44ral|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^a5yqph44ral


>%%
>```annotation-json
>{"created":"2023-11-14T17:37:23.368Z","updated":"2023-11-14T17:37:23.368Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":14364,"end":14519},{"type":"TextQuoteSelector","exact":" Asin previous works, we thus seek a soft partial assignmentmatrix P ∈ [0,1]M×N between local features in A and B,from which we can extract correspondences","prefix":" occlusion or non-repeatability.","suffix":".Overview – Figure 3: LightGlue "}]}]}
>```
>%%
>*%%PREFIX%%occlusion or non-repeatability.%%HIGHLIGHT%% ==Asin previous works, we thus seek a soft partial assignmentmatrix P ∈ [0,1]M×N between local features in A and B,from which we can extract correspondences== %%POSTFIX%%.Overview – Figure 3: LightGlue*
>%%LINK%%[[#^kg1k04m8cq|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kg1k04m8cq


>%%
>```annotation-json
>{"created":"2023-11-14T17:47:22.218Z","updated":"2023-11-14T17:47:22.218Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":14954,"end":15027},{"type":"TextQuoteSelector","exact":"We associate each local feature i in image I ∈ {A,B}with a state xIi ∈ Rd","prefix":"ations.3.1. Transformer backbone","suffix":". The state is initialized with "}]}]}
>```
>%%
>*%%PREFIX%%ations.3.1. Transformer backbone%%HIGHLIGHT%% ==We associate each local feature i in image I ∈ {A,B}with a state xIi ∈ Rd== %%POSTFIX%%. The state is initialized with*
>%%LINK%%[[#^ueainw0khs|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^ueainw0khs


>%%
>```annotation-json
>{"created":"2023-11-14T17:52:21.186Z","updated":"2023-11-14T17:52:21.186Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":13960,"end":13980},{"type":"TextQuoteSelector","exact":"pi := (x,y)i ∈[0,1]2","prefix":"composed of a 2D point position ","suffix":", nor-malized by the image size,"}]}]}
>```
>%%
>*%%PREFIX%%composed of a 2D point position%%HIGHLIGHT%% ==pi := (x,y)i ∈[0,1]2== %%POSTFIX%%, nor-malized by the image size,*
>%%LINK%%[[#^gbs8urvd07h|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^gbs8urvd07h


>%%
>```annotation-json
>{"created":"2023-11-14T17:53:01.996Z","updated":"2023-11-14T17:53:01.996Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":15369,"end":15404},{"type":"TextQuoteSelector","exact":"xIi ←xIi + MLP ([xIi |mI←Si]) , (1)","prefix":"tedfrom a source image S ∈{A,B}:","suffix":"where [·|·] stacks two vectors. "}]}]}
>```
>%%
>*%%PREFIX%%tedfrom a source image S ∈{A,B}:%%HIGHLIGHT%% ==xIi ←xIi + MLP ([xIi |mI←Si]) , (1)== %%POSTFIX%%where [·|·] stacks two vectors.*
>%%LINK%%[[#^8u4lqbs5yzn|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^8u4lqbs5yzn


>%%
>```annotation-json
>{"created":"2023-11-14T17:56:41.866Z","updated":"2023-11-14T17:56:41.866Z","document":{"title":"LightGlue.pdf","link":[{"href":"urn:x-pdf:07153b1aa79a51ab5125ae2e6a3044dc"},{"href":"vault:/Sources/Image Matching/LightGlue.pdf"}],"documentFingerprint":"07153b1aa79a51ab5125ae2e6a3044dc"},"uri":"vault:/Sources/Image Matching/LightGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/LightGlue.pdf","selector":[{"type":"TextPositionSelector","start":15790,"end":16004},{"type":"TextQuoteSelector","exact":"mI←Si = ∑j∈SSoftmaxk∈S(aISik)j WxSj , (2)where W is a projection matrix and aISij is an attention scorebetween points i and j of images I and S. How this score iscomputed differs for self- and cross-attention units","prefix":"rage of all states j of image S:","suffix":".Self-attention: Each point atte"}]}]}
>```
>%%
>*%%PREFIX%%rage of all states j of image S:%%HIGHLIGHT%% ==mI←Si = ∑j∈SSoftmaxk∈S(aISik)j WxSj , (2)where W is a projection matrix and aISij is an attention scorebetween points i and j of images I and S. How this score iscomputed differs for self- and cross-attention units== %%POSTFIX%%.Self-attention: Each point atte*
>%%LINK%%[[#^yyv9295zdwp|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^yyv9295zdwp
