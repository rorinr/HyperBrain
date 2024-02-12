annotation-target::SuperPoint.pdf

>%%
>```annotation-json
>{"created":"2023-09-30T11:30:48.481Z","updated":"2023-09-30T11:30:48.481Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":1443,"end":1571},{"type":"TextQuoteSelector","exact":"In-terest points are 2D locations in an image which are stableand repeatable from different lighting conditions and view-points.","prefix":"ct interest points from images. ","suffix":" The subfield of mathematics and"}]}]}
>```
>%%
>*%%PREFIX%%ct interest points from images.%%HIGHLIGHT%% ==In-terest points are 2D locations in an image which are stableand repeatable from different lighting conditions and view-points.== %%POSTFIX%%The subfield of mathematics and*
>%%LINK%%[[#^15mifke3nke|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>#definition
^15mifke3nke


>%%
>```annotation-json
>{"created":"2023-09-30T11:37:29.492Z","text":"When you train a CNN for a task, having well-defined and consistent labels is crucial. In the case of human-body keypoint estimation, you can easily obtain labeled data where the keypoints correspond to anatomical features. However, for interest point detection, obtaining consistent and meaningful labels can be problematic because there's no universally agreed-upon definition of what constitutes an \"interest point.\"","updated":"2023-09-30T11:37:29.492Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":2695,"end":3241},{"type":"TextQuoteSelector","exact":"It seems natural to similarly formulate interest point de-tection as a large-scale supervised machine learning prob-lem and train the latest convolutional neural network ar-chitecture to detect them. Unfortunately, when comparedto semantic tasks such as human-body keypoint estimation,where a network is trained to detect body parts such as thecorner of the mouth or left ankle, the notion of interest pointdetection is semantically ill-defined. Thus training convo-lution neural networks with strong supervision of interestpoints is non-trivial.","prefix":"ons labeled by human annotators.","suffix":"Instead of using human supervisi"}]}]}
>```
>%%
>*%%PREFIX%%ons labeled by human annotators.%%HIGHLIGHT%% ==It seems natural to similarly formulate interest point de-tection as a large-scale supervised machine learning prob-lem and train the latest convolutional neural network ar-chitecture to detect them. Unfortunately, when comparedto semantic tasks such as human-body keypoint estimation,where a network is trained to detect body parts such as thecorner of the mouth or left ankle, the notion of interest pointdetection is semantically ill-defined. Thus training convo-lution neural networks with strong supervision of interestpoints is non-trivial.== %%POSTFIX%%Instead of using human supervisi*
>%%LINK%%[[#^luwvkqtpcp|show annotation]]
>%%COMMENT%%
>When you train a CNN for a task, having well-defined and consistent labels is crucial. In the case of human-body keypoint estimation, you can easily obtain labeled data where the keypoints correspond to anatomical features. However, for interest point detection, obtaining consistent and meaningful labels can be problematic because there's no universally agreed-upon definition of what constitutes an "interest point."
>%%TAGS%%
>#image matching
^luwvkqtpcp


>%%
>```annotation-json
>{"created":"2023-09-30T11:42:01.924Z","updated":"2023-09-30T11:42:01.924Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":3377,"end":3580},{"type":"TextQuoteSelector","exact":"In our approach, we create a largedataset of pseudo-ground truth interest point locations inreal images, supervised by the interest point detector itself,rather than a large-scale human annotation effort","prefix":" solu-tion using self-training. ","suffix":".To generate the pseudo-ground t"}]}]}
>```
>%%
>*%%PREFIX%%solu-tion using self-training.%%HIGHLIGHT%% ==In our approach, we create a largedataset of pseudo-ground truth interest point locations inreal images, supervised by the interest point detector itself,rather than a large-scale human annotation effort== %%POSTFIX%%.To generate the pseudo-ground t*
>%%LINK%%[[#^3e5fk2jbz8d|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^3e5fk2jbz8d


>%%
>```annotation-json
>{"created":"2023-09-30T11:44:48.685Z","updated":"2023-09-30T11:44:48.685Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":4558,"end":4717},{"type":"TextQuoteSelector","exact":"The synthetic dataset con-sists of simple geometric shapes with no ambiguity in theinterest point locations. We call the resulting trained de-tector MagicPoint","prefix":"-thetic Shapes (see Figure 2a). ","suffix":"—it significantly outperforms tr"}]}]}
>```
>%%
>*%%PREFIX%%-thetic Shapes (see Figure 2a).%%HIGHLIGHT%% ==The synthetic dataset con-sists of simple geometric shapes with no ambiguity in theinterest point locations. We call the resulting trained de-tector MagicPoint== %%POSTFIX%%—it significantly outperforms tr*
>%%LINK%%[[#^kpm9bd45md|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kpm9bd45md


>%%
>```annotation-json
>{"created":"2023-09-30T11:53:32.967Z","updated":"2023-09-30T11:53:32.967Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":8892,"end":9100},{"type":"TextQuoteSelector","exact":" fully-convolutional neural network archi-tecture called SuperPoint which operates on a full-sized im-age and produces interest point detections accompanied byfixed length descriptors in a single forward pass","prefix":"rPoint ArchitectureWe designed a","suffix":" (see Fig-ure 3). The model has "}]}]}
>```
>%%
>*%%PREFIX%%rPoint ArchitectureWe designed a%%HIGHLIGHT%% ==fully-convolutional neural network archi-tecture called SuperPoint which operates on a full-sized im-age and produces interest point detections accompanied byfixed length descriptors in a single forward pass== %%POSTFIX%%(see Fig-ure 3). The model has*
>%%LINK%%[[#^f5e4lriah6f|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^f5e4lriah6f



>%%
>```annotation-json
>{"created":"2023-09-30T12:15:17.489Z","text":"Interest point decoder produces probability of distinctiveness of each pixel","updated":"2023-09-30T12:15:17.489Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":10352,"end":10480},{"type":"TextQuoteSelector","exact":"For interest point detection, each pixel of the output cor-responds to a probability of “point-ness” for that pixel in theinput.","prefix":"> 1).3.2. Interest Point Decoder","suffix":" The standard network design for"}]}]}
>```
>%%
>*%%PREFIX%%> 1).3.2. Interest Point Decoder%%HIGHLIGHT%% ==For interest point detection, each pixel of the output cor-responds to a probability of “point-ness” for that pixel in theinput.== %%POSTFIX%%The standard network design for*
>%%LINK%%[[#^mknqhk0ho2|show annotation]]
>%%COMMENT%%
>Interest point decoder produces probability of distinctiveness of each pixel
>%%TAGS%%
>
^mknqhk0ho2



>%%
>```annotation-json
>{"created":"2023-09-30T12:19:37.205Z","text":"PixelShuffle in InterestPointDecoder to avoid unwanted checkerboard effects and ","updated":"2023-09-30T12:19:37.205Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":10873,"end":10989},{"type":"TextQuoteSelector","exact":"thuswe designed the interest point detection head with an ex-plicit decoder1 to reduce the computation of the model.","prefix":"ed checkerboard artifacts [18], ","suffix":"The interest point detector head"}]}]}
>```
>%%
>*%%PREFIX%%ed checkerboard artifacts [18],%%HIGHLIGHT%% ==thuswe designed the interest point detection head with an ex-plicit decoder1 to reduce the computation of the model.== %%POSTFIX%%The interest point detector head*
>%%LINK%%[[#^sj1a0amr55|show annotation]]
>%%COMMENT%%
>PixelShuffle in InterestPointDecoder to avoid unwanted checkerboard effects and 
>%%TAGS%%
>
^sj1a0amr55


>%%
>```annotation-json
>{"created":"2023-09-30T13:05:20.121Z","updated":"2023-09-30T13:05:20.121Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":12204,"end":12393},{"type":"TextQuoteSelector","exact":"The decoder then performs bi-cubic interpolation of the descriptor and then L2-normalizesthe activations to be unit length. This fixed, non-learned de-scriptor decoder is shown in Figure 3.","prefix":"ndkeeps the run-time tractable. ","suffix":"3.4. Loss FunctionsThe final los"}]}]}
>```
>%%
>*%%PREFIX%%ndkeeps the run-time tractable.%%HIGHLIGHT%% ==The decoder then performs bi-cubic interpolation of the descriptor and then L2-normalizesthe activations to be unit length. This fixed, non-learned de-scriptor decoder is shown in Figure 3.== %%POSTFIX%%3.4. Loss FunctionsThe final los*
>%%LINK%%[[#^z7ye5ri89n|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^z7ye5ri89n



>%%
>```annotation-json
>{"created":"2023-10-16T04:36:44.011Z","updated":"2023-10-16T04:36:44.011Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":491,"end":655},{"type":"TextQuoteSelector","exact":"our fully-convolutional model operates on full-sizedimages and jointly computes pixel-level interest point loca-tions and associated descriptors in one forward pass","prefix":"o patch-based neural net-works, ","suffix":". Weintroduce Homographic Adapta"}]}]}
>```
>%%
>*%%PREFIX%%o patch-based neural net-works,%%HIGHLIGHT%% ==our fully-convolutional model operates on full-sizedimages and jointly computes pixel-level interest point loca-tions and associated descriptors in one forward pass== %%POSTFIX%%. Weintroduce Homographic Adapta*
>%%LINK%%[[#^hnhnz6tmo96|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^hnhnz6tmo96


>%%
>```annotation-json
>{"created":"2023-10-16T05:00:10.522Z","updated":"2023-10-16T05:00:10.522Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":9118,"end":9394},{"type":"TextQuoteSelector","exact":"The model has a single, shared encoder to pro-cess and reduce the input image dimensionality. After theencoder, the architecture splits into two decoder “heads”,which learn task specific weights – one for interest point de-tection and the other for interest point description.","prefix":"e forward pass (see Fig-ure 3). ","suffix":" Most ofthe network’s parameters"}]}]}
>```
>%%
>*%%PREFIX%%e forward pass (see Fig-ure 3).%%HIGHLIGHT%% ==The model has a single, shared encoder to pro-cess and reduce the input image dimensionality. After theencoder, the architecture splits into two decoder “heads”,which learn task specific weights – one for interest point de-tection and the other for interest point description.== %%POSTFIX%%Most ofthe network’s parameters*
>%%LINK%%[[#^jrfz02k4ta8|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^jrfz02k4ta8


>%%
>```annotation-json
>{"created":"2023-10-16T05:14:10.098Z","updated":"2023-10-16T05:14:10.098Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":10989,"end":11305},{"type":"TextQuoteSelector","exact":"The interest point detector head computes X ∈RHc×Wc×65 and outputs a tensor sized RH×W. The 65channels correspond to local, non-overlapping 8 × 8 gridregions of pixels plus an extra “no interest point” dustbin.After a channel-wise softmax, the dustbin dimension is re-moved and a RHc×Wc×64 ⇒RH×W reshape is performed","prefix":"ce the computation of the model.","suffix":".1 This decoder has no parameter"}]}]}
>```
>%%
>*%%PREFIX%%ce the computation of the model.%%HIGHLIGHT%% ==The interest point detector head computes X ∈RHc×Wc×65 and outputs a tensor sized RH×W. The 65channels correspond to local, non-overlapping 8 × 8 gridregions of pixels plus an extra “no interest point” dustbin.After a channel-wise softmax, the dustbin dimension is re-moved and a RHc×Wc×64 ⇒RH×W reshape is performed== %%POSTFIX%%.1 This decoder has no parameter*
>%%LINK%%[[#^r5jyvm4gtta|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^r5jyvm4gtta


>%%
>```annotation-json
>{"created":"2023-10-16T05:29:54.925Z","text":"Softmax of the 8x8 cells + one dustbin scalar if there is not keypoint at all","updated":"2023-10-16T05:29:54.925Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":12954,"end":13396},{"type":"TextQuoteSelector","exact":"The interest point detector loss function Lp is a fully-convolutional cross-entropy loss over the cells xhw ∈ X.We call the set of corresponding ground-truth interest pointlabels2 Y and individual entries as yhw. The loss is:Lp(X,Y ) = 1HcWcHc,Wc∑h=1w=1lp(xhw; yhw), (2)wherelp(xhw; y) = −log(exp(xhwy)∑65k=1 exp(xhwk)). (3)2 If two ground truth corner positions land in the same bin then we ran-domly select one ground truth corner location.","prefix":" + Lp(X′,Y ′) + λLd(D,D′,S). (1)","suffix":"3Train MagicPoint Base DetectorM"}]}]}
>```
>%%
>*%%PREFIX%%+ Lp(X′,Y ′) + λLd(D,D′,S). (1)%%HIGHLIGHT%% ==The interest point detector loss function Lp is a fully-convolutional cross-entropy loss over the cells xhw ∈ X.We call the set of corresponding ground-truth interest pointlabels2 Y and individual entries as yhw. The loss is:Lp(X,Y ) = 1HcWcHc,Wc∑h=1w=1lp(xhw; yhw), (2)wherelp(xhw; y) = −log(exp(xhwy)∑65k=1 exp(xhwk)). (3)2 If two ground truth corner positions land in the same bin then we ran-domly select one ground truth corner location.== %%POSTFIX%%3Train MagicPoint Base DetectorM*
>%%LINK%%[[#^opmov2dn4e|show annotation]]
>%%COMMENT%%
>Softmax of the 8x8 cells + one dustbin scalar if there is not keypoint at all
>%%TAGS%%
>
^opmov2dn4e


>%%
>```annotation-json
>{"created":"2023-10-16T05:45:15.987Z","updated":"2023-10-16T05:45:15.987Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":13846,"end":14467},{"type":"TextQuoteSelector","exact":"The descriptor loss is applied to all pairs of descriptorcells, dhw ∈ D from the first image and d′h′w′∈ D′from the second image. The homography-induced corre-spondence between the (h,w) cell and the (h′,w′) cell canbe written as follows:shwh′w′={1, if ||̂Hphw −ph′w′||≤80, otherwise (4)where phw denotes the location of the center pixel in the(h,w) cell, and ̂Hphw denotes multiplying the cell locationphw by the homography H and dividing by the last coor-dinate, as is usually done when transforming between Eu-clidean and homogeneous coordinates. We denote the entireset of correspondences for a pair of images with S.","prefix":"compared to classical detectors.","suffix":"We also add a weighting term λd "}]}]}
>```
>%%
>*%%PREFIX%%compared to classical detectors.%%HIGHLIGHT%% ==The descriptor loss is applied to all pairs of descriptorcells, dhw ∈ D from the first image and d′h′w′∈ D′from the second image. The homography-induced corre-spondence between the (h,w) cell and the (h′,w′) cell canbe written as follows:shwh′w′={1, if ||̂Hphw −ph′w′||≤80, otherwise (4)where phw denotes the location of the center pixel in the(h,w) cell, and ̂Hphw denotes multiplying the cell locationphw by the homography H and dividing by the last coor-dinate, as is usually done when transforming between Eu-clidean and homogeneous coordinates. We denote the entireset of correspondences for a pair of images with S.== %%POSTFIX%%We also add a weighting term λd*
>%%LINK%%[[#^n4rjrcq0op|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^n4rjrcq0op


>%%
>```annotation-json
>{"created":"2023-10-16T05:49:59.053Z","updated":"2023-10-16T05:49:59.053Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":14467,"end":14828},{"type":"TextQuoteSelector","exact":"We also add a weighting term λd to help balance the factthat there are more negative correspondences than positiveones. We use a hinge loss with positive margin mp andnegative margin mn. The descriptor loss is defined as:Ld(D,D′,S) =1(HcWc)2Hc,Wc∑h=1w=1Hc,Wc∑h′=1w′=1ld(dhw,d′h′w′; shwh′w′), (5)whereld(d,d′; s) = λd ∗s ∗max(0,mp −dTd′)+(1 −s) ∗max(0,dTd′ −mn).","prefix":"ces for a pair of images with S.","suffix":" (6)4. Synthetic Pre-TrainingIn "}]}]}
>```
>%%
>*%%PREFIX%%ces for a pair of images with S.%%HIGHLIGHT%% ==We also add a weighting term λd to help balance the factthat there are more negative correspondences than positiveones. We use a hinge loss with positive margin mp andnegative margin mn. The descriptor loss is defined as:Ld(D,D′,S) =1(HcWc)2Hc,Wc∑h=1w=1Hc,Wc∑h′=1w′=1ld(dhw,d′h′w′; shwh′w′), (5)whereld(d,d′; s) = λd ∗s ∗max(0,mp −dTd′)+(1 −s) ∗max(0,dTd′ −mn).== %%POSTFIX%%(6)4. Synthetic Pre-TrainingIn*
>%%LINK%%[[#^78k5dhr8a34|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^78k5dhr8a34


>%%
>```annotation-json
>{"created":"2023-10-16T07:46:13.103Z","updated":"2023-10-16T07:46:13.103Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":18323,"end":18878},{"type":"TextQuoteSelector","exact":"Our system bootstraps itself from a base interest pointdetector and a large set of unlabeled images from the targetdomain (e.g., MS-COCO). Operating in a self-supervisedparadigm (also known as self-training), we first generate aset of pseudo-ground truth interest point locations for eachimage in the target domain, then use traditional supervisedlearning machinery. At the core of our method is a processthat applies random homographies to warped copies of theinput image and combines the results – a process we callHomographic Adaptation (see Figure 5).","prefix":"tation.5. Homographic Adaptation","suffix":"5.1. FormulationHomographies giv"}]}]}
>```
>%%
>*%%PREFIX%%tation.5. Homographic Adaptation%%HIGHLIGHT%% ==Our system bootstraps itself from a base interest pointdetector and a large set of unlabeled images from the targetdomain (e.g., MS-COCO). Operating in a self-supervisedparadigm (also known as self-training), we first generate aset of pseudo-ground truth interest point locations for eachimage in the target domain, then use traditional supervisedlearning machinery. At the core of our method is a processthat applies random homographies to warped copies of theinput image and combines the results – a process we callHomographic Adaptation (see Figure 5).== %%POSTFIX%%5.1. FormulationHomographies giv*
>%%LINK%%[[#^1nijx1gjz8g|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^1nijx1gjz8g


>%%
>```annotation-json
>{"created":"2023-10-16T07:50:34.042Z","updated":"2023-10-16T07:50:34.042Z","document":{"title":"SuperPoint.pdf","link":[{"href":"urn:x-pdf:8e6b66da0c023627666c42b9986be078"},{"href":"vault:/Sources/Image Matching/SuperPoint.pdf"}],"documentFingerprint":"8e6b66da0c023627666c42b9986be078"},"uri":"vault:/Sources/Image Matching/SuperPoint.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperPoint.pdf","selector":[{"type":"TextPositionSelector","start":23714,"end":23943},{"type":"TextQuoteSelector","exact":"Figure 7. Iterative Homographic Adaptation. Top row: ini-tial base detector (MagicPoint) struggles to find repeatable de-tections. Middle and bottom rows: further training with Homo-graphic Adaption improves detector performance.","prefix":"MagicPointHomographic Adaptation","suffix":"model is trained for 200,000 ite"}]}]}
>```
>%%
>*%%PREFIX%%MagicPointHomographic Adaptation%%HIGHLIGHT%% ==Figure 7. Iterative Homographic Adaptation. Top row: ini-tial base detector (MagicPoint) struggles to find repeatable de-tections. Middle and bottom rows: further training with Homo-graphic Adaption improves detector performance.== %%POSTFIX%%model is trained for 200,000 ite*
>%%LINK%%[[#^kx8fyiyhjli|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^kx8fyiyhjli
