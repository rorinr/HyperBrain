annotation-target::D2Net.pdf

>%%
>```annotation-json
>{"created":"2023-09-22T10:53:32.137Z","updated":"2023-09-22T10:53:32.137Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":480,"end":806},{"type":"TextQuoteSelector","exact":"propose an approach where a single convolu-tional neural network plays a dual role: It is simultane-ously a dense feature descriptor and a feature detector.By postponing the detection to a later stage, the obtainedkeypoints are more stable than their traditional counter-parts based on early detection of low-level structures.","prefix":"fficult imaging condi-tions. We ","suffix":" Weshow that this model can be t"}]}]}
>```
>%%
>*%%PREFIX%%fficult imaging condi-tions. We%%HIGHLIGHT%% ==propose an approach where a single convolu-tional neural network plays a dual role: It is simultane-ously a dense feature descriptor and a feature detector.By postponing the detection to a later stage, the obtainedkeypoints are more stable than their traditional counter-parts based on early detection of low-level structures.== %%POSTFIX%%Weshow that this model can be t*
>%%LINK%%[[#^bg1gqdejmec|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^bg1gqdejmec


>%%
>```annotation-json
>{"created":"2023-09-22T10:56:56.040Z","text":"Gap between the structures to find a keypoint vs the structures to find the corresponding descriptor","updated":"2023-09-22T10:56:56.040Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":3491,"end":3643},{"type":"TextQuoteSelector","exact":"While localdescriptors consider larger patches and potentially encodehigher-level structures, the keypoint detector only consid-ers small image regions.","prefix":"ility in the keypoint detector: ","suffix":" As a result, the detections are"}]}]}
>```
>%%
>*%%PREFIX%%ility in the keypoint detector:%%HIGHLIGHT%% ==While localdescriptors consider larger patches and potentially encodehigher-level structures, the keypoint detector only consid-ers small image regions.== %%POSTFIX%%As a result, the detections are*
>%%LINK%%[[#^kwt4mat1g1|show annotation]]
>%%COMMENT%%
>Gap between the structures to find a keypoint vs the structures to find the corresponding descriptor
>%%TAGS%%
>
^kwt4mat1g1


>%%
>```annotation-json
>{"created":"2023-09-22T15:09:47.709Z","updated":"2023-09-22T15:09:47.709Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":4054,"end":4189},{"type":"TextQuoteSelector","exact":"Thus, approaches that forego thedetection stage and instead densely extract descriptors per-form much better in challenging conditions.","prefix":"ectedreliably [46, 59, 62, 71]. ","suffix":" Yet, this gainin robustness com"}]}]}
>```
>%%
>*%%PREFIX%%ectedreliably [46, 59, 62, 71].%%HIGHLIGHT%% ==Thus, approaches that forego thedetection stage and instead densely extract descriptors per-form much better in challenging conditions.== %%POSTFIX%%Yet, this gainin robustness com*
>%%LINK%%[[#^sj84cca2i7|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^sj84cca2i7


>%%
>```annotation-json
>{"created":"2023-09-22T15:10:58.898Z","updated":"2023-09-22T15:10:58.898Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":4559,"end":4947},{"type":"TextQuoteSelector","exact":"Rather than performingfeature detection early on based on low-level information,we propose to postpone the detection stage. We first com-pute a set of feature maps via a Deep Convolutional NeuralNetwork (CNN). These feature maps are then used to com-pute the descriptors (as slices through all maps at a specificpixel position) and to detect keypoints (as local maxima ofthe feature maps)","prefix":"ture detection and description: ","suffix":". As a result, the feature detec"}]}]}
>```
>%%
>*%%PREFIX%%ture detection and description:%%HIGHLIGHT%% ==Rather than performingfeature detection early on based on low-level information,we propose to postpone the detection stage. We first com-pute a set of feature maps via a Deep Convolutional NeuralNetwork (CNN). These feature maps are then used to com-pute the descriptors (as slices through all maps at a specificpixel position) and to detect keypoints (as local maxima ofthe feature maps)== %%POSTFIX%%. As a result, the feature detec*
>%%LINK%%[[#^bbn0zd2ggm8|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^bbn0zd2ggm8


>%%
>```annotation-json
>{"created":"2023-09-22T15:43:31.229Z","updated":"2023-09-22T15:43:31.229Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":6588,"end":7017},{"type":"TextQuoteSelector","exact":"The most common approach to sparse fea-ture extraction – the detect-then-describe approach – firstperforms feature detection [7, 19, 30, 32, 34] and then ex-tracts a feature descriptor [7, 9, 25, 30, 45] from a patchcentered around each keypoint. The keypoint detector istypically responsible for providing robustness or invarianceagainst effects such as scale, rotation, or viewpoint changesby normalizing the patch accordingly.","prefix":".2. Related WorkLocal features. ","suffix":" However, some ofthese responsib"}]}]}
>```
>%%
>*%%PREFIX%%.2. Related WorkLocal features.%%HIGHLIGHT%% ==The most common approach to sparse fea-ture extraction – the detect-then-describe approach – firstperforms feature detection [7, 19, 30, 32, 34] and then ex-tracts a feature descriptor [7, 9, 25, 30, 45] from a patchcentered around each keypoint. The keypoint detector istypically responsible for providing robustness or invarianceagainst effects such as scale, rotation, or viewpoint changesby normalizing the patch accordingly.== %%POSTFIX%%However, some ofthese responsib*
>%%LINK%%[[#^gthzcmzuxfv|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^gthzcmzuxfv


>%%
>```annotation-json
>{"created":"2023-09-22T15:45:53.649Z","updated":"2023-09-22T15:45:53.649Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":7394,"end":7652},{"type":"TextQuoteSelector","exact":"For ef-ficiency, the feature detector often considers only small im-age regions [65] and typically focuses on low-level struc-tures such as corners [19] or blobs [30]. The descriptorthen captures higher level information in a larger patcharound the keypoint.","prefix":"tector and descriptor [39, 65]. ","suffix":" In contrast, this paper propose"}]}]}
>```
>%%
>*%%PREFIX%%tector and descriptor [39, 65].%%HIGHLIGHT%% ==For ef-ficiency, the feature detector often considers only small im-age regions [65] and typically focuses on low-level struc-tures such as corners [19] or blobs [30]. The descriptorthen captures higher level information in a larger patcharound the keypoint.== %%POSTFIX%%In contrast, this paper propose*
>%%LINK%%[[#^9r3qhh6pqq|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^9r3qhh6pqq


>%%
>```annotation-json
>{"created":"2023-09-22T15:47:03.152Z","updated":"2023-09-22T15:47:03.152Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":7795,"end":7905},{"type":"TextQuoteSelector","exact":"our approachis able to detect keypoints belonging to higher-level struc-tures and locally unique descriptors. ","prefix":" shown in Fig. 2b. As a result, ","suffix":"The work closest toour approach "}]}]}
>```
>%%
>*%%PREFIX%%shown in Fig. 2b. As a result,%%HIGHLIGHT%% ==our approachis able to detect keypoints belonging to higher-level struc-tures and locally unique descriptors.== %%POSTFIX%%The work closest toour approach*
>%%LINK%%[[#^phvgdyw8it|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^phvgdyw8it


>%%
>```annotation-json
>{"created":"2023-09-23T08:50:45.144Z","updated":"2023-09-23T08:50:45.144Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":11197,"end":11336},{"type":"TextQuoteSelector","exact":"input image I to obtain a 3D tensor F = F(I),F ∈Rh×w×n, where h×w is the spatial resolution of the featuremaps and n the number of channels","prefix":"ethod is to apply a CNN F onthe ","suffix":".3.1. Feature DescriptionAs in o"}]}]}
>```
>%%
>*%%PREFIX%%ethod is to apply a CNN F onthe%%HIGHLIGHT%% ==input image I to obtain a 3D tensor F = F(I),F ∈Rh×w×n, where h×w is the spatial resolution of the featuremaps and n the number of channels== %%POSTFIX%%.3.1. Feature DescriptionAs in o*
>%%LINK%%[[#^ypojs2u8phs|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^ypojs2u8phs


>%%
>```annotation-json
>{"created":"2023-09-23T08:51:23.158Z","text":"Interpretation: They produced a embedding space of H,W,N (similar to images) where each \"pixel\" is one descriptor, ie one descriptor consists of N scalars","updated":"2023-09-23T08:51:23.158Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":11361,"end":11965},{"type":"TextQuoteSelector","exact":"As in other previous work [38, 44, 59], the most straight-forward interpretation of the 3D tensor F is as a dense setof descriptor vectors d:dij = Fij:,d∈Rn , (1)with i = 1,...,h and j = 1,...,w. These descriptor vec-tors can be readily compared between images to establishcorrespondences using the Euclidean distance. During thetraining stage, these descriptors will be adjusted such thatthe same points in the scene produce similar descriptors,even when the images contain strong appearance changes.In practice, we apply an L2 normalization on the descriptorsprior to comparing them: ˆdij = dij/‖dij‖2.","prefix":"hannels.3.1. Feature Description","suffix":"3.2. Feature DetectionA differen"}]}]}
>```
>%%
>*%%PREFIX%%hannels.3.1. Feature Description%%HIGHLIGHT%% ==As in other previous work [38, 44, 59], the most straight-forward interpretation of the 3D tensor F is as a dense setof descriptor vectors d:dij = Fij:,d∈Rn , (1)with i = 1,...,h and j = 1,...,w. These descriptor vec-tors can be readily compared between images to establishcorrespondences using the Euclidean distance. During thetraining stage, these descriptors will be adjusted such thatthe same points in the scene produce similar descriptors,even when the images contain strong appearance changes.In practice, we apply an L2 normalization on the descriptorsprior to comparing them: ˆdij = dij/‖dij‖2.== %%POSTFIX%%3.2. Feature DetectionA differen*
>%%LINK%%[[#^jj3174hecr|show annotation]]
>%%COMMENT%%
>Interpretation: They produced a embedding space of H,W,N (similar to images) where each "pixel" is one descriptor, ie one descriptor consists of N scalars
>%%TAGS%%
>
^jj3174hecr


>%%
>```annotation-json
>{"created":"2023-09-23T08:54:37.494Z","updated":"2023-09-23T08:54:37.494Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":12119,"end":12427},{"type":"TextQuoteSelector","exact":"In this interpretation, the feature ex-traction function F can be thought of as n different featuredetector functions Dk, each producing a 2D response mapDk. These detection response maps are analogous to theDifference-of-Gaussians (DoG) response maps obtained inScale Invariant Feature Transform (SIFT) [30]","prefix":"k ∈Rh×w , (2)where k = 1,...,n. ","suffix":" or to the cor-nerness score map"}]}]}
>```
>%%
>*%%PREFIX%%k ∈Rh×w , (2)where k = 1,...,n.%%HIGHLIGHT%% ==In this interpretation, the feature ex-traction function F can be thought of as n different featuredetector functions Dk, each producing a 2D response mapDk. These detection response maps are analogous to theDifference-of-Gaussians (DoG) response maps obtained inScale Invariant Feature Transform (SIFT) [30]== %%POSTFIX%%or to the cor-nerness score map*
>%%LINK%%[[#^lrz1o49txbs|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^lrz1o49txbs



>%%
>```annotation-json
>{"created":"2023-09-23T09:02:15.268Z","text":"Requirements for finding a detection at pixel (i,j): Given n different feature detectors. Select the feature detector k  where (i,j) is most preeminent. Iff there is a local-maximum at (i,j) on that particular detectors response map D_k, then (i,j) is a detection","updated":"2023-09-23T09:02:15.268Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":12807,"end":13345},{"type":"TextQuoteSelector","exact":"How-ever, in our approach, contrary to traditional feature detec-tors, there exist multiple detection maps Dk (k = 1,...,n),and a detection can take place on any of them. Therefore,for a point (i,j) to be detected, we require:(i,j) is a detection ⇐⇒Dkij is a local max. in Dk ,with k = arg maxtDtij .(3)Intuitively, for each pixel (i,j), this corresponds to select-ing the most preeminent detector Dk (channel selection),and then verifying whether there is a local-maximum at po-sition (i,j) on that particular detector’s response map Dk.","prefix":" non-local-maximum suppression. ","suffix":"featureextractionsoft detection "}]}]}
>```
>%%
>*%%PREFIX%%non-local-maximum suppression.%%HIGHLIGHT%% ==How-ever, in our approach, contrary to traditional feature detec-tors, there exist multiple detection maps Dk (k = 1,...,n),and a detection can take place on any of them. Therefore,for a point (i,j) to be detected, we require:(i,j) is a detection ⇐⇒Dkij is a local max. in Dk ,with k = arg maxtDtij .(3)Intuitively, for each pixel (i,j), this corresponds to select-ing the most preeminent detector Dk (channel selection),and then verifying whether there is a local-maximum at po-sition (i,j) on that particular detector’s response map Dk.== %%POSTFIX%%featureextractionsoft detection*
>%%LINK%%[[#^jw2qg4ryb9a|show annotation]]
>%%COMMENT%%
>Requirements for finding a detection at pixel (i,j): Given n different feature detectors. Select the feature detector k  where (i,j) is most preeminent. Iff there is a local-maximum at (i,j) on that particular detectors response map D_k, then (i,j) is a detection
>%%TAGS%%
>
^jw2qg4ryb9a


>%%
>```annotation-json
>{"created":"2023-09-23T09:08:13.713Z","text":"Compute softmax score of each single pixel in the whole HxWxN feature space with its 9 neighbors","updated":"2023-09-23T09:08:13.713Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":14146,"end":14313},{"type":"TextQuoteSelector","exact":"First, we define a soft local-max. scoreαkij = exp (Dkij)∑(i′,j′)∈N(i,j) exp(Dki′j′) , (4)where N(i,j) is the set of 9 neighbours of the pixel (i,j)(including itself).","prefix":"e amenable forback-propagation. ","suffix":" Then, we define the soft channe"}]}]}
>```
>%%
>*%%PREFIX%%e amenable forback-propagation.%%HIGHLIGHT%% ==First, we define a soft local-max. scoreαkij = exp (Dkij)∑(i′,j′)∈N(i,j) exp(Dki′j′) , (4)where N(i,j) is the set of 9 neighbours of the pixel (i,j)(including itself).== %%POSTFIX%%Then, we define the soft channe*
>%%LINK%%[[#^l5m84x9kpfk|show annotation]]
>%%COMMENT%%
>Compute softmax score of each single pixel in the whole HxWxN feature space with its 9 neighbors
>%%TAGS%%
>
^l5m84x9kpfk


>%%
>```annotation-json
>{"created":"2023-09-23T09:14:11.398Z","text":"Divides each scalar by the maximum in the same descriptor (where descriptor refers to the vector of one pixel (i,j) with n scalars)","updated":"2023-09-23T09:14:11.398Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":14455,"end":14477},{"type":"TextQuoteSelector","exact":"βkij = Dkij/maxt Dtij ","prefix":"el-wise non-maximum suppression:","suffix":". (5)Next, in order to take both"}]}]}
>```
>%%
>*%%PREFIX%%el-wise non-maximum suppression:%%HIGHLIGHT%% ==βkij = Dkij/maxt Dtij== %%POSTFIX%%. (5)Next, in order to take both*
>%%LINK%%[[#^chfaomy68wf|show annotation]]
>%%COMMENT%%
>Divides each scalar by the maximum in the same descriptor (where descriptor refers to the vector of one pixel (i,j) with n scalars)
>%%TAGS%%
>
^chfaomy68wf


>%%
>```annotation-json
>{"created":"2023-09-23T09:29:56.750Z","text":"What do they mean by both criteria?\n- alpha is the soft local max, ie the score of a pixel in its neighborhood of the same detection map k\n- beta is a score of a scalar wrt to the scalars in the same descriptor\n- gamma now multiplies each alpha of a pixel with the corresponding beta of the same pixel and chooses the maximum of the products along the whole descriptor of on pixel\n!!! Note that we reduce the feature space here to one single channel. Before we had N channels","updated":"2023-09-23T09:29:56.750Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":14482,"end":14646},{"type":"TextQuoteSelector","exact":"Next, in order to take both criteria into account, we maxi-mize the product of both scores across all feature maps k toobtain a single score map:γij = maxk(αkijβkij","prefix":"sion:βkij = Dkij/maxt Dtij . (5)","suffix":") . (6)Finally, the soft detecti"}]}]}
>```
>%%
>*%%PREFIX%%sion:βkij = Dkij/maxt Dtij . (5)%%HIGHLIGHT%% ==Next, in order to take both criteria into account, we maxi-mize the product of both scores across all feature maps k toobtain a single score map:γij = maxk(αkijβkij== %%POSTFIX%%) . (6)Finally, the soft detecti*
>%%LINK%%[[#^99ouabx4us7|show annotation]]
>%%COMMENT%%
>What do they mean by both criteria?
>- alpha is the soft local max, ie the score of a pixel in its neighborhood of the same detection map k
>- beta is a score of a scalar wrt to the scalars in the same descriptor
>- gamma now multiplies each alpha of a pixel with the corresponding beta of the same pixel and chooses the maximum of the products along the whole descriptor of on pixel
>!!! Note that we reduce the feature space here to one single channel. Before we had N channels
>%%TAGS%%
>
^99ouabx4us7


>%%
>```annotation-json
>{"created":"2023-09-27T07:55:46.735Z","updated":"2023-09-27T07:55:46.735Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":14925,"end":15055},{"type":"TextQuoteSelector","exact":"they are not inherently invariant toscale changes and the matching tends to fail in cases with asignificant difference in viewpoin","prefix":"raining withdata augmentations, ","suffix":"t.In order to obtain features th"}]}]}
>```
>%%
>*%%PREFIX%%raining withdata augmentations,%%HIGHLIGHT%% ==they are not inherently invariant toscale changes and the matching tends to fail in cases with asignificant difference in viewpoin== %%POSTFIX%%t.In order to obtain features th*
>%%LINK%%[[#^prxr5788abm|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^prxr5788abm


>%%
>```annotation-json
>{"created":"2023-09-27T07:58:43.828Z","updated":"2023-09-27T07:58:43.828Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":15272,"end":15534},{"type":"TextQuoteSelector","exact":"This is only per-formed during test time.Given the input image I, an image pyramid Iρ contain-ing three different resolutions ρ = 0.5,1,2 (correspondingto half resolution, input resolution, and double resolution)is constructed and used to extract feature maps Fρ","prefix":"for some object detectors [16]. ","suffix":" at eachresolution. Then, the la"}]}]}
>```
>%%
>*%%PREFIX%%for some object detectors [16].%%HIGHLIGHT%% ==This is only per-formed during test time.Given the input image I, an image pyramid Iρ contain-ing three different resolutions ρ = 0.5,1,2 (correspondingto half resolution, input resolution, and double resolution)is constructed and used to extract feature maps Fρ== %%POSTFIX%%at eachresolution. Then, the la*
>%%LINK%%[[#^a39th4e305|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^a39th4e305


>%%
>```annotation-json
>{"created":"2023-09-27T08:05:25.815Z","updated":"2023-09-27T08:05:25.815Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":16079,"end":16280},{"type":"TextQuoteSelector","exact":"Starting at the coarsest scale, we markthe detected positions; these masks are upsampled (nearestneighbor) to the resolutions of the next scales; detectionsfalling into marked regions are then ignored.","prefix":"owing responsegating mechanism: ","suffix":"4. Jointly optimizing detection "}]}]}
>```
>%%
>*%%PREFIX%%owing responsegating mechanism:%%HIGHLIGHT%% ==Starting at the coarsest scale, we markthe detected positions; these masks are upsampled (nearestneighbor) to the resolutions of the next scales; detectionsfalling into marked regions are then ignored.== %%POSTFIX%%4. Jointly optimizing detection*
>%%LINK%%[[#^8d5lrm8p71p|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^8d5lrm8p71p


>%%
>```annotation-json
>{"created":"2023-09-27T08:05:31.312Z","text":"Is that one really a summation of images? How does this work?","updated":"2023-09-27T08:05:31.312Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":15696,"end":15712},{"type":"TextQuoteSelector","exact":"Fρ = Fρ + ∑γ<ρFγ","prefix":"on ones, in the following way: ̃","suffix":" . (8)Note that the feature maps"}]}]}
>```
>%%
>*%%PREFIX%%on ones, in the following way: ̃%%HIGHLIGHT%% ==Fρ = Fρ + ∑γ<ρFγ== %%POSTFIX%%. (8)Note that the feature maps*
>%%LINK%%[[#^hm3gb4cum09|show annotation]]
>%%COMMENT%%
>Is that one really a summation of images? How does this work?
>%%TAGS%%
>
^hm3gb4cum09


>%%
>```annotation-json
>{"created":"2023-09-28T07:45:05.853Z","text":"Standard l2 norm for positive descriptors: These should be as close to each other as possible","updated":"2023-09-28T07:45:05.853Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":17562,"end":17661},{"type":"TextQuoteSelector","exact":"positive descriptor distance p(c) between thecorresponding descriptors as:p(c) = ‖ˆd(1)A −ˆd(2)B ‖2","prefix":"ures. To this end,we define the ","suffix":" , (9)The negative distance n(c)"}]}]}
>```
>%%
>*%%PREFIX%%ures. To this end,we define the%%HIGHLIGHT%% ==positive descriptor distance p(c) between thecorresponding descriptors as:p(c) = ‖ˆd(1)A −ˆd(2)B ‖2== %%POSTFIX%%, (9)The negative distance n(c)*
>%%LINK%%[[#^hsjlc55k7cc|show annotation]]
>%%COMMENT%%
>Standard l2 norm for positive descriptors: These should be as close to each other as possible
>%%TAGS%%
>
^hsjlc55k7cc



>%%
>```annotation-json
>{"created":"2023-09-28T08:09:13.790Z","text":"d_N1 or d_N2 are these descriptors which are the hardest for the model to find: These are the descriptors with the minimal l2_norm, but which lie outside of the neighborhood of the correct correspondence (where K is the threshold of how 'large' this neighborhood is).","updated":"2023-09-28T08:09:13.790Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":17667,"end":18044},{"type":"TextQuoteSelector","exact":"The negative distance n(c), which accounts for the mostconfounding descriptor for either ˆd(1)A or ˆd(2)B , is defined as:n(c) = min(‖ˆd(1)A −ˆd(2)N2‖2,‖ˆd(1)N1 −ˆd(2)B ‖2), (10)where the negative samples d(1)N1 and d(2)N2 are the hardestnegatives that lie outside of a square local neighbourhoodof the correct correspondence:N1 = arg minP∈I1‖ˆd(1)P −ˆd(2)B ‖2 s.t. ‖P −A‖∞ > K","prefix":":p(c) = ‖ˆd(1)A −ˆd(2)B ‖2 , (9)","suffix":" , (11)and similarly for N2. The"}]}]}
>```
>%%
>*%%PREFIX%%:p(c) = ‖ˆd(1)A −ˆd(2)B ‖2 , (9)%%HIGHLIGHT%% ==The negative distance n(c), which accounts for the mostconfounding descriptor for either ˆd(1)A or ˆd(2)B , is defined as:n(c) = min(‖ˆd(1)A −ˆd(2)N2‖2,‖ˆd(1)N1 −ˆd(2)B ‖2), (10)where the negative samples d(1)N1 and d(2)N2 are the hardestnegatives that lie outside of a square local neighbourhoodof the correct correspondence:N1 = arg minP∈I1‖ˆd(1)P −ˆd(2)B ‖2 s.t. ‖P −A‖∞ > K== %%POSTFIX%%, (11)and similarly for N2. The*
>%%LINK%%[[#^oxvzusenz4f|show annotation]]
>%%COMMENT%%
>d_N1 or d_N2 are these descriptors which are the hardest for the model to find: These are the descriptors with the minimal l2_norm, but which lie outside of the neighborhood of the correct correspondence (where K is the threshold of how 'large' this neighborhood is).
>%%TAGS%%
>
^oxvzusenz4f


>%%
>```annotation-json
>{"created":"2023-09-28T08:13:11.369Z","updated":"2023-09-28T08:13:11.369Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":18142,"end":18364},{"type":"TextQuoteSelector","exact":"m(c) = max (0,M + p(c)2 −n(c)2) . (12)Intuitively, this triplet margin ranking loss seeks to enforcethe distinctiveness of descriptors by penalizing any con-founding descriptor that would lead to a wrong match as-signment.","prefix":"margin M can be then defined as:","suffix":" In order to additionally seek f"}]}]}
>```
>%%
>*%%PREFIX%%margin M can be then defined as:%%HIGHLIGHT%% ==m(c) = max (0,M + p(c)2 −n(c)2) . (12)Intuitively, this triplet margin ranking loss seeks to enforcethe distinctiveness of descriptors by penalizing any con-founding descriptor that would lead to a wrong match as-signment.== %%POSTFIX%%In order to additionally seek f*
>%%LINK%%[[#^h9evp8ohdkk|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^h9evp8ohdkk


>%%
>```annotation-json
>{"created":"2023-09-28T12:21:45.924Z","text":"ATM: Unclear how this fraction works.","updated":"2023-09-28T12:21:45.924Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":18365,"end":18569},{"type":"TextQuoteSelector","exact":"In order to additionally seek for the repeatabilityof detections, an detection term is added to the triplet mar-gin ranking loss in the following way:L(I1,I2) = ∑c∈Cs(1)c s(2)c∑q∈C s(1)q s(2)qm(p(c),n(c))","prefix":"d to a wrong match as-signment. ","suffix":" , (13)where s(1)c and s(2)c are"}]}]}
>```
>%%
>*%%PREFIX%%d to a wrong match as-signment.%%HIGHLIGHT%% ==In order to additionally seek for the repeatabilityof detections, an detection term is added to the triplet mar-gin ranking loss in the following way:L(I1,I2) = ∑c∈Cs(1)c s(2)c∑q∈C s(1)q s(2)qm(p(c),n(c))== %%POSTFIX%%, (13)where s(1)c and s(2)c are*
>%%LINK%%[[#^upd8f3qo26s|show annotation]]
>%%COMMENT%%
>ATM: Unclear how this fraction works.
>%%TAGS%%
>
^upd8f3qo26s


>%%
>```annotation-json
>{"created":"2023-09-28T12:24:42.364Z","updated":"2023-09-28T12:24:42.364Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":20553,"end":20638},{"type":"TextQuoteSelector","exact":"For each pair, we selected a ran-dom 256 ×256 crop centered around one correspondence","prefix":"mbalancepresent in the dataset. ","suffix":".We use a batch size of 1 and ma"}]}]}
>```
>%%
>*%%PREFIX%%mbalancepresent in the dataset.%%HIGHLIGHT%% ==For each pair, we selected a ran-dom 256 ×256 crop centered around one correspondence== %%POSTFIX%%.We use a batch size of 1 and ma*
>%%LINK%%[[#^nx506av469n|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^nx506av469n



>%%
>```annotation-json
>{"created":"2023-09-28T12:28:47.567Z","text":"Possible baseline for thesis aswell?","updated":"2023-09-28T12:28:47.567Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":24358,"end":24754},{"type":"TextQuoteSelector","exact":"As baselines for the classical detect-then-describe strat-egy, we use RootSIFT [4, 30] with the Hessian Affine key-point detector [32], a variant using a learned shape estimator(HesAffNet [36] - HAN) and descriptor (HardNet++ [35] -HN++2), and an end-to-end trainable variant (LF-Net [39]).We also compare against SuperPoint [13] and DELF [38],which are conceptually more similar to our approach.","prefix":" correct matches per image pair.","suffix":"Results. Fig. 4 shows results fo"}]}]}
>```
>%%
>*%%PREFIX%%correct matches per image pair.%%HIGHLIGHT%% ==As baselines for the classical detect-then-describe strat-egy, we use RootSIFT [4, 30] with the Hessian Affine key-point detector [32], a variant using a learned shape estimator(HesAffNet [36] - HAN) and descriptor (HardNet++ [35] -HN++2), and an end-to-end trainable variant (LF-Net [39]).We also compare against SuperPoint [13] and DELF [38],which are conceptually more similar to our approach.== %%POSTFIX%%Results. Fig. 4 shows results fo*
>%%LINK%%[[#^ui3nu5x752|show annotation]]
>%%COMMENT%%
>Possible baseline for thesis aswell?
>%%TAGS%%
>
^ui3nu5x752


>%%
>```annotation-json
>{"created":"2023-09-28T12:31:13.515Z","text":"Suitable for medial images?","updated":"2023-09-28T12:31:13.515Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":25861,"end":26223},{"type":"TextQuoteSelector","exact":"As can be expected, our method performs worsethan detect-then-describe approaches for stricter matchingthresholds: The latter use detectors firing at low-level blob-like structures, which are inherently better localized thanthe higher-level features used by our approach. At the sametime, our features are also detected at the lower resolutionof the CNN features","prefix":"uited for this type of matching.","suffix":".We suspect that the inferior pe"}]}]}
>```
>%%
>*%%PREFIX%%uited for this type of matching.%%HIGHLIGHT%% ==As can be expected, our method performs worsethan detect-then-describe approaches for stricter matchingthresholds: The latter use detectors firing at low-level blob-like structures, which are inherently better localized thanthe higher-level features used by our approach. At the sametime, our features are also detected at the lower resolutionof the CNN features== %%POSTFIX%%.We suspect that the inferior pe*
>%%LINK%%[[#^9f5b87g8ygt|show annotation]]
>%%COMMENT%%
>Suitable for medial images?
>%%TAGS%%
>
^9f5b87g8ygt


>%%
>```annotation-json
>{"created":"2023-10-19T09:39:50.249Z","updated":"2023-10-19T09:39:50.249Z","document":{"title":"D2Net.pdf","link":[{"href":"urn:x-pdf:bd70fdf474f05ed6c5670fe7d8719a63"},{"href":"vault:/Sources/Image Matching/D2Net.pdf"}],"documentFingerprint":"bd70fdf474f05ed6c5670fe7d8719a63"},"uri":"vault:/Sources/Image Matching/D2Net.pdf","target":[{"source":"vault:/Sources/Image Matching/D2Net.pdf","selector":[{"type":"TextPositionSelector","start":851,"end":973},{"type":"TextQuoteSelector","exact":"pixel correspon-dences extracted from readily available large-scale SfM re-constructions, without any further annotations.","prefix":"this model can be trained using ","suffix":" The pro-posed method obtains st"}]}]}
>```
>%%
>*%%PREFIX%%this model can be trained using%%HIGHLIGHT%% ==pixel correspon-dences extracted from readily available large-scale SfM re-constructions, without any further annotations.== %%POSTFIX%%The pro-posed method obtains st*
>%%LINK%%[[#^uarad735kq|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^uarad735kq
