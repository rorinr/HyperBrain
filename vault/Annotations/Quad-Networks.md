annotation-target::QuadNetworks.pdf

>%%
>```annotation-json
>{"created":"2023-10-03T10:47:23.117Z","text":"Ranking is invariant to augmentations which allows to find point in the transformed image again.","updated":"2023-10-03T10:47:23.117Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":4112,"end":4565},{"type":"TextQuoteSelector","exact":"The idea of our method is to train a neural network thatmaps an object point to a single real-valued response andthen rank points according to this response. This rankingis optimized to be repeatable under the desired transforma-tion classes: if one point is higher in the ranking than an-other one, it should still be higher after a transformation.Consequently, the top/bottom quantiles of the response arerepeatable and can be used as interest points.","prefix":"arns the solu-tion from scratch.","suffix":" This idea isillustrated in Fig."}]}]}
>```
>%%
>*%%PREFIX%%arns the solu-tion from scratch.%%HIGHLIGHT%% ==The idea of our method is to train a neural network thatmaps an object point to a single real-valued response andthen rank points according to this response. This rankingis optimized to be repeatable under the desired transforma-tion classes: if one point is higher in the ranking than an-other one, it should still be higher after a transformation.Consequently, the top/bottom quantiles of the response arerepeatable and can be used as interest points.== %%POSTFIX%%This idea isillustrated in Fig.*
>%%LINK%%[[#^z67pqurhlp|show annotation]]
>%%COMMENT%%
>Ranking is invariant to augmentations which allows to find point in the transformed image again.
>%%TAGS%%
>
^z67pqurhlp


>%%
>```annotation-json
>{"created":"2023-10-03T19:10:54.635Z","text":"Synthesize labels by transforming the image","updated":"2023-10-03T19:10:54.635Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":9005,"end":9460},{"type":"TextQuoteSelector","exact":"In this section we introduce the problem of learningan interest point detector as the problem of learning torank points. We consider interest points to come fromthe top/bottom quantiles of some response function. Ifthese quantiles are preserved under certain transformationclasses, we have a good detector: it re-detects the samepoints. For the quantiles of the ranking to be preserved,we search for a ranking which is invariant to those transfor-mations.","prefix":"itypairs.3. Detection by ranking","suffix":"Let us consider a set D of objec"}]}]}
>```
>%%
>*%%PREFIX%%itypairs.3. Detection by ranking%%HIGHLIGHT%% ==In this section we introduce the problem of learningan interest point detector as the problem of learning torank points. We consider interest points to come fromthe top/bottom quantiles of some response function. Ifthese quantiles are preserved under certain transformationclasses, we have a good detector: it re-detects the samepoints. For the quantiles of the ranking to be preserved,we search for a ranking which is invariant to those transfor-mations.== %%POSTFIX%%Let us consider a set D of objec*
>%%LINK%%[[#^wzs64kz0dza|show annotation]]
>%%COMMENT%%
>Synthesize labels by transforming the image
>%%TAGS%%
>
^wzs64kz0dza


>%%
>```annotation-json
>{"created":"2023-10-03T19:13:39.946Z","text":"H() returns a scalar, which allows us to sort the outputs of H. Formula 1 makes sure that this order is the same for 2 different images (one original image and a transformed version of it)","updated":"2023-10-03T19:13:39.946Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":10765,"end":11033},{"type":"TextQuoteSelector","exact":"Thus, invarianceof the ranking under transformation t ∈ T can be stated asfollows: for every quadruple (pid,pjd,pit(d),pjt(d)) satisfyingi,j ∈Cdt,i 6= j, it holds thatH(pid|w) > H(pjd|w) & H(pit(d)|w) > H(pjt(d)|w)orH(pid|w) < H(pjd|w) & H(pit(d)|w) < H(pjt(d)|w)","prefix":"oice of H is a neural network). ","suffix":" .(1)From the condition above it"}]}]}
>```
>%%
>*%%PREFIX%%oice of H is a neural network).%%HIGHLIGHT%% ==Thus, invarianceof the ranking under transformation t ∈ T can be stated asfollows: for every quadruple (pid,pjd,pit(d),pjt(d)) satisfyingi,j ∈Cdt,i 6= j, it holds thatH(pid|w) > H(pjd|w) & H(pit(d)|w) > H(pjt(d)|w)orH(pid|w) < H(pjd|w) & H(pit(d)|w) < H(pjt(d)|w)== %%POSTFIX%%.(1)From the condition above it*
>%%LINK%%[[#^sxyk6wst1s|show annotation]]
>%%COMMENT%%
>H() returns a scalar, which allows us to sort the outputs of H. Formula 1 makes sure that this order is the same for 2 different images (one original image and a transformed version of it)
>%%TAGS%%
>
^sxyk6wst1s


>%%
>```annotation-json
>{"created":"2023-10-03T19:16:41.447Z","updated":"2023-10-03T19:16:41.447Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":11408,"end":11456},{"type":"TextQuoteSelector","exact":"take the top/bottom quantiles as interest points","prefix":"t d by their responseH(p|w) and ","suffix":".In the next section, we will st"}]}]}
>```
>%%
>*%%PREFIX%%t d by their responseH(p|w) and%%HIGHLIGHT%% ==take the top/bottom quantiles as interest points== %%POSTFIX%%.In the next section, we will st*
>%%LINK%%[[#^0n254ys06jp|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^0n254ys06jp


>%%
>```annotation-json
>{"created":"2023-10-03T19:19:42.995Z","text":"The difference in the brackets must have the same sign to satisfy inequality (3)","updated":"2023-10-03T19:19:42.995Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":11663,"end":11833},{"type":"TextQuoteSelector","exact":"R(pid,pjd,pit(d),pjt(d)|w) =(H(pid|w) −H(pjd|w))(H(pit(d)|w) −H(pjt(d)|w)) .(2)Then the ranking invariance condition (1) can be re-writtenasR(pid,pjd,pit(d),pjt(d)|w) > 0","prefix":"greement function forquadruples:","suffix":" . (3)In order to give preferenc"}]}]}
>```
>%%
>*%%PREFIX%%greement function forquadruples:%%HIGHLIGHT%% ==R(pid,pjd,pit(d),pjt(d)|w) =(H(pid|w) −H(pjd|w))(H(pit(d)|w) −H(pjt(d)|w)) .(2)Then the ranking invariance condition (1) can be re-writtenasR(pid,pjd,pit(d),pjt(d)|w) > 0== %%POSTFIX%%. (3)In order to give preferenc*
>%%LINK%%[[#^appnmq8r8mu|show annotation]]
>%%COMMENT%%
>The difference in the brackets must have the same sign to satisfy inequality (3)
>%%TAGS%%
>
^appnmq8r8mu


>%%
>```annotation-json
>{"created":"2023-10-03T19:39:38.307Z","text":"Sparsity means in this context to only select a limited number of points, while ensuring that these are highly distinctive/informative","updated":"2023-10-03T19:39:38.307Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":13501,"end":13743},{"type":"TextQuoteSelector","exact":"It is typical for interest point detectors to ensure spar-sity in two ways: by retaining the top/bottom quantiles ofthe response function (contrast filtering) and by retainingthe local extrema of the response function (non-maximumsuppression)","prefix":"ages observing the same 3Dscene.","suffix":". While Observation 1 suggests r"}]}]}
>```
>%%
>*%%PREFIX%%ages observing the same 3Dscene.%%HIGHLIGHT%% ==It is typical for interest point detectors to ensure spar-sity in two ways: by retaining the top/bottom quantiles ofthe response function (contrast filtering) and by retainingthe local extrema of the response function (non-maximumsuppression)== %%POSTFIX%%. While Observation 1 suggests r*
>%%LINK%%[[#^u9hwm7fbxe|show annotation]]
>%%COMMENT%%
>Sparsity means in this context to only select a limited number of points, while ensuring that these are highly distinctive/informative
>%%TAGS%%
>
^u9hwm7fbxe


>%%
>```annotation-json
>{"created":"2023-10-06T06:37:57.394Z","text":"unsupervised approach creates labels by augmentation (more a self supervised one)","updated":"2023-10-06T06:37:57.394Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":16795,"end":16913},{"type":"TextQuoteSelector","exact":"a fully-unsupervised RGB detector (correspondencesare obtained by randomly warping images and chang-ing illumination),","prefix":"scanned 3D pointsonto images),• ","suffix":"• a cross-modal RGB/depth detect"}]}]}
>```
>%%
>*%%PREFIX%%scanned 3D pointsonto images),•%%HIGHLIGHT%% ==a fully-unsupervised RGB detector (correspondencesare obtained by randomly warping images and chang-ing illumination),== %%POSTFIX%%• a cross-modal RGB/depth detect*
>%%LINK%%[[#^4464h9guqzs|show annotation]]
>%%COMMENT%%
>unsupervised approach creates labels by augmentation (more a self supervised one)
>%%TAGS%%
>
^4464h9guqzs


>%%
>```annotation-json
>{"created":"2023-10-06T06:42:18.092Z","updated":"2023-10-06T06:42:18.092Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":20378,"end":20545},{"type":"TextQuoteSelector","exact":"he patches are preprocessed as it istypical for neural networks: the mean over the whole patchis subtracted, then it is divided by the standard deviationover the patch","prefix":"or, weconvert it to grayscale. T","suffix":".Augmentation. We augment the tr"}]}]}
>```
>%%
>*%%PREFIX%%or, weconvert it to grayscale. T%%HIGHLIGHT%% ==he patches are preprocessed as it istypical for neural networks: the mean over the whole patchis subtracted, then it is divided by the standard deviationover the patch== %%POSTFIX%%.Augmentation. We augment the tr*
>%%LINK%%[[#^kuxe31e87|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kuxe31e87


>%%
>```annotation-json
>{"created":"2023-10-06T06:43:47.157Z","updated":"2023-10-06T06:43:47.157Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":22197,"end":22382},{"type":"TextQuoteSelector","exact":"Our learned model detectspoints different from DoG: they are more evenly distributedin images. That is usually profitable for estimating geomet-ric transformations between camera frames","prefix":" incomparison to DoG in Fig. 5. ","suffix":".The learned response functions "}]}]}
>```
>%%
>*%%PREFIX%%incomparison to DoG in Fig. 5.%%HIGHLIGHT%% ==Our learned model detectspoints different from DoG: they are more evenly distributedin images. That is usually profitable for estimating geomet-ric transformations between camera frames== %%POSTFIX%%.The learned response functions*
>%%LINK%%[[#^i8m95s4rem|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^i8m95s4rem


>%%
>```annotation-json
>{"created":"2023-10-06T06:44:47.133Z","text":"True positive rate","updated":"2023-10-06T06:44:47.133Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":22838,"end":22939},{"type":"TextQuoteSelector","exact":"For that we use the same matching score asin [22], i.e., the ratio of correct matches to all matches.","prefix":"detected points can be5matched. ","suffix":" Ourdetectors (Linear, Non-linea"}]}]}
>```
>%%
>*%%PREFIX%%detected points can be5matched.%%HIGHLIGHT%% ==For that we use the same matching score asin [22], i.e., the ratio of correct matches to all matches.== %%POSTFIX%%Ourdetectors (Linear, Non-linea*
>%%LINK%%[[#^wryk7cfeajj|show annotation]]
>%%COMMENT%%
>True positive rate
>%%TAGS%%
>
^wryk7cfeajj


>%%
>```annotation-json
>{"created":"2023-10-06T06:46:11.765Z","text":"self supervised","updated":"2023-10-06T06:46:11.765Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":23415,"end":23682},{"type":"TextQuoteSelector","exact":"The goal of this experiment is to show that ground-truthcorrespondences from an additional data source (like 3Dpoints from a laser scanner) are not necessary to train a de-tector with our method. Instead, we can sample randomtransformations to obtain correspondences.","prefix":" Fully-unsupervised RGB detector","suffix":"Training. In this experiment, we"}]}]}
>```
>%%
>*%%PREFIX%%Fully-unsupervised RGB detector%%HIGHLIGHT%% ==The goal of this experiment is to show that ground-truthcorrespondences from an additional data source (like 3Dpoints from a laser scanner) are not necessary to train a de-tector with our method. Instead, we can sample randomtransformations to obtain correspondences.== %%POSTFIX%%Training. In this experiment, we*
>%%LINK%%[[#^n6z7yoe3juj|show annotation]]
>%%COMMENT%%
>self supervised
>%%TAGS%%
>
^n6z7yoe3juj


>%%
>```annotation-json
>{"created":"2023-10-06T06:48:42.137Z","text":"What does repeatability exactly mean? ","updated":"2023-10-06T06:48:42.137Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":27269,"end":27365},{"type":"TextQuoteSelector","exact":"The repeatability and filters from the best model(Deep Conv Net) are shown in Fig. 6 and Fig. 8.","prefix":"(f(32,32),e)8,f(32,1)).Results. ","suffix":" Our bestmodel outperformes othe"}]}]}
>```
>%%
>*%%PREFIX%%(f(32,32),e)8,f(32,1)).Results.%%HIGHLIGHT%% ==The repeatability and filters from the best model(Deep Conv Net) are shown in Fig. 6 and Fig. 8.== %%POSTFIX%%Our bestmodel outperformes othe*
>%%LINK%%[[#^y7ljsx1e0pd|show annotation]]
>%%COMMENT%%
>What does repeatability exactly mean? 
>%%TAGS%%
>#question
^y7ljsx1e0pd


>%%
>```annotation-json
>{"created":"2023-10-06T07:31:15.152Z","text":"Hinge loss","updated":"2023-10-06T07:31:15.152Z","document":{"title":"QuadNetworks.pdf","link":[{"href":"urn:x-pdf:1113effdeb4bf0efd914a0b909bd8c87"},{"href":"vault:/Sources/QuadNetworks.pdf"}],"documentFingerprint":"1113effdeb4bf0efd914a0b909bd8c87"},"uri":"vault:/Sources/QuadNetworks.pdf","target":[{"source":"vault:/Sources/QuadNetworks.pdf","selector":[{"type":"TextPositionSelector","start":12450,"end":12585},{"type":"TextQuoteSelector","exact":"`(R) = max(0,1 −R) . (6)Then the final form of our minimized objective will beL(w)= ∑d∈D∑t∈T∑i,j∈Cdtmax(0,1−R(pid,pjd,pit(d),pjt(d)|w))","prefix":" we choose to use the hinge loss","suffix":",(7)which is differentiable as l"}]}]}
>```
>%%
>*%%PREFIX%%we choose to use the hinge loss%%HIGHLIGHT%% ==`(R) = max(0,1 −R) . (6)Then the final form of our minimized objective will beL(w)= ∑d∈D∑t∈T∑i,j∈Cdtmax(0,1−R(pid,pjd,pit(d),pjt(d)|w))== %%POSTFIX%%,(7)which is differentiable as l*
>%%LINK%%[[#^zxoqajygwj8|show annotation]]
>%%COMMENT%%
>Hinge loss
>%%TAGS%%
>
^zxoqajygwj8
