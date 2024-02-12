annotation-target::DISK.pdf


>%%
>```annotation-json
>{"created":"2023-09-29T08:19:02.877Z","text":"Challenges in image matching","updated":"2023-09-29T08:19:02.877Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":1840,"end":2210},{"type":"TextQuoteSelector","exact":"Given two images A and B with feature sets FA and FB, matching them is O(|FA|·|FB|).As each image pixel may become a feature, the problem quickly becomes intractable. Moreover, the“quality” of a given feature depends on the rest, because a feature that is very similar to others is lessdistinctive, and therefore less useful. This is hard to account for during training.","prefix":"to its computationalcomplexity. ","suffix":"We address this issue by bridgin"}]}]}
>```
>%%
>*%%PREFIX%%to its computationalcomplexity.%%HIGHLIGHT%% ==Given two images A and B with feature sets FA and FB, matching them is O(|FA|·|FB|).As each image pixel may become a feature, the problem quickly becomes intractable. Moreover, the“quality” of a given feature depends on the rest, because a feature that is very similar to others is lessdistinctive, and therefore less useful. This is hard to account for during training.== %%POSTFIX%%We address this issue by bridgin*
>%%LINK%%[[#^o53a3mha1nf|show annotation]]
>%%COMMENT%%
>Challenges in image matching
>%%TAGS%%
>#challenges
^o53a3mha1nf


>%%
>```annotation-json
>{"created":"2023-09-29T08:34:31.850Z","text":"1. Subdivide keypoint-channel into a grid (hxh)\n2. Select one (at most) point per grid cell:\n2.1 normalize the grid using softmax\n2.2 sample from this softmax probability dist.","updated":"2023-09-29T08:34:31.850Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":8344,"end":8697},{"type":"TextQuoteSelector","exact":"The detection map K is subdivided into a grid with cell size h ×h, and we select at most onefeature per grid cell, similarly to SuperPoint [10]. To do so, we crop the feature map correspondingto cell u, denoted Ku, and use a softmax operator to normalize it. Our probabilistic frameworksamples a pixel p in cell u with probability Ps(p|Ku) = softmax(Ku)","prefix":"iptors [20, 25, 21, 40, 13, 31].","suffix":"p. This detection proposalp may "}]}]}
>```
>%%
>*%%PREFIX%%iptors [20, 25, 21, 40, 13, 31].%%HIGHLIGHT%% ==The detection map K is subdivided into a grid with cell size h ×h, and we select at most onefeature per grid cell, similarly to SuperPoint [10]. To do so, we crop the feature map correspondingto cell u, denoted Ku, and use a softmax operator to normalize it. Our probabilistic frameworksamples a pixel p in cell u with probability Ps(p|Ku) = softmax(Ku)== %%POSTFIX%%p. This detection proposalp may*
>%%LINK%%[[#^3hprw8oeta7|show annotation]]
>%%COMMENT%%
>1. Subdivide keypoint-channel into a grid (hxh)
>2. Select one (at most) point per grid cell:
>2.1 normalize the grid using softmax
>2.2 sample from this softmax probability dist.
>%%TAGS%%
>
^3hprw8oeta7


>%%
>```annotation-json
>{"created":"2023-09-29T08:37:58.639Z","text":"Use the selected keypoint from above with a probability of sigmoid(x), where x is the scalar value of the original pixel in the detection/keypoint channel, before softmax (afaik)","updated":"2023-09-29T08:37:58.639Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":8700,"end":8901},{"type":"TextQuoteSelector","exact":"This detection proposalp may still be rejected: we accept it with probability Pa(acceptp|Ku) = σ(Kup), where Kup isthe (scalar) value of the detection map K at location p in cell u, and σ is a sigmoid.","prefix":"bility Ps(p|Ku) = softmax(Ku)p. ","suffix":" Note thatPs(p|Ku) models relati"}]}]}
>```
>%%
>*%%PREFIX%%bility Ps(p|Ku) = softmax(Ku)p.%%HIGHLIGHT%% ==This detection proposalp may still be rejected: we accept it with probability Pa(acceptp|Ku) = σ(Kup), where Kup isthe (scalar) value of the detection map K at location p in cell u, and σ is a sigmoid.== %%POSTFIX%%Note thatPs(p|Ku) models relati*
>%%LINK%%[[#^ci8p0wbslyw|show annotation]]
>%%COMMENT%%
>Use the selected keypoint from above with a probability of sigmoid(x), where x is the scalar value of the original pixel in the detection/keypoint channel, before softmax (afaik)
>%%TAGS%%
>
^ci8p0wbslyw


>%%
>```annotation-json
>{"created":"2023-09-29T08:39:30.285Z","text":"- P_s(p|K^u) is relative preference, because we compare each value of the detection-channel with its neighborhood\n- P_a(accept_p|K^u) is absolute quality if the location, because we directly use the scalar in the detection-channel","updated":"2023-09-29T08:39:30.285Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":8902,"end":9049},{"type":"TextQuoteSelector","exact":"Note thatPs(p|Ku) models relative preference across a set of different locations, whereas Pa(acceptp|Ku)models the absolute quality for location p.","prefix":" in cell u, and σ is a sigmoid. ","suffix":" The total probability of sampli"}]}]}
>```
>%%
>*%%PREFIX%%in cell u, and σ is a sigmoid.%%HIGHLIGHT%% ==Note thatPs(p|Ku) models relative preference across a set of different locations, whereas Pa(acceptp|Ku)models the absolute quality for location p.== %%POSTFIX%%The total probability of sampli*
>%%LINK%%[[#^sglam0623pb|show annotation]]
>%%COMMENT%%
>- P_s(p|K^u) is relative preference, because we compare each value of the detection-channel with its neighborhood
>- P_a(accept_p|K^u) is absolute quality if the location, because we directly use the scalar in the detection-channel
>%%TAGS%%
>
^sglam0623pb



>%%
>```annotation-json
>{"created":"2023-09-29T08:51:36.862Z","text":"as in lecture","updated":"2023-09-29T08:51:36.862Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":10184,"end":10381},{"type":"TextQuoteSelector","exact":"The ratio test, introduced by SIFT [20], rejects a match if the ratio of the distancesbetween its first and second nearest neighbours is above a threshold, in order to only return confidentmatches.","prefix":"asing the ratioof correct ones. ","suffix":" These two approaches are often "}]}]}
>```
>%%
>*%%PREFIX%%asing the ratioof correct ones.%%HIGHLIGHT%% ==The ratio test, introduced by SIFT [20], rejects a match if the ratio of the distancesbetween its first and second nearest neighbours is above a threshold, in order to only return confidentmatches.== %%POSTFIX%%These two approaches are often*
>%%LINK%%[[#^s233uh7f3i|show annotation]]
>%%COMMENT%%
>as in lecture
>%%TAGS%%
>
^s233uh7f3i


>%%
>```annotation-json
>{"created":"2023-09-29T09:29:18.459Z","text":"The descriptors need to be nearest neighbours of ech other!","updated":"2023-09-29T09:29:18.459Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":9984,"end":10183},{"type":"TextQuoteSelector","exact":" Cycle-consistent matching enforces that two features be nearest neighbours of eachother in descriptor space, cutting down on the number of putative matches while increasing the ratioof correct ones.","prefix":"tent matchingand the ratio test.","suffix":" The ratio test, introduced by S"}]}]}
>```
>%%
>*%%PREFIX%%tent matchingand the ratio test.%%HIGHLIGHT%% ==Cycle-consistent matching enforces that two features be nearest neighbours of eachother in descriptor space, cutting down on the number of putative matches while increasing the ratioof correct ones.== %%POSTFIX%%The ratio test, introduced by S*
>%%LINK%%[[#^ifdlpc6rhvj|show annotation]]
>%%COMMENT%%
>The descriptors need to be nearest neighbours of ech other!
>%%TAGS%%
>
^ifdlpc6rhvj


>%%
>```annotation-json
>{"created":"2023-09-29T09:32:42.352Z","text":"AFAIK:\n- instead of just assigning the nearest neighbors/descriptors to each other, we sample using the distances (l2)\n- we want to sample more frequently from closer descriptors\n-> softmax assigns higher probabilities to higher scores. Since better matches yield in smaller l2-distances, while worse matches in larger, we must multiply the distance with a negative scalar to make sure smaller l2-distance matches are preferred, ie yielding in higher softmax-probability","updated":"2023-09-29T09:32:42.352Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":10553,"end":11132},{"type":"TextQuoteSelector","exact":"Our solution is to relax cycle-consistent matching. Conceptually, we draw forward (A\u0001B) matchesfor features FA,i from categorical distributions defined by the rows of distance matrix d, and reverse(A\u0000B) matches for features FB,j from distributions based on its columns. We declare FA,i to matchFB,j if both the forward and reverse matches are sampled, i.e., if the samples are consistent. Theforward distribution of matches is given by PA\u0001B(j|d,i) = softmax (−θMd(i,·))j, where θM isthe single parameter, the inverse of the softmax temperature. PA\u0000B is analogously defined by dT.","prefix":"y are not easily differentiable.","suffix":"It should be noted that, given f"}]}]}
>```
>%%
>*%%PREFIX%%y are not easily differentiable.%%HIGHLIGHT%% ==Our solution is to relax cycle-consistent matching. Conceptually, we draw forward (AB) matchesfor features FA,i from categorical distributions defined by the rows of distance matrix d, and reverse(A B) matches for features FB,j from distributions based on its columns. We declare FA,i to matchFB,j if both the forward and reverse matches are sampled, i.e., if the samples are consistent. Theforward distribution of matches is given by PAB(j|d,i) = softmax (−θMd(i,·))j, where θM isthe single parameter, the inverse of the softmax temperature. PA B is analogously defined by dT.== %%POSTFIX%%It should be noted that, given f*
>%%LINK%%[[#^hrsruqachjs|show annotation]]
>%%COMMENT%%
>AFAIK:
>- instead of just assigning the nearest neighbors/descriptors to each other, we sample using the distances (l2)
>- we want to sample more frequently from closer descriptors
>-> softmax assigns higher probabilities to higher scores. Since better matches yield in smaller l2-distances, while worse matches in larger, we must multiply the distance with a negative scalar to make sure smaller l2-distance matches are preferred, ie yielding in higher softmax-probability
>%%TAGS%%
>
^hrsruqachjs


>%%
>```annotation-json
>{"created":"2023-09-29T09:48:32.736Z","text":"Since the distance matrix d is symmetric (since l2-distance is), arent P_A->B and P_B->A the same?","updated":"2023-09-29T09:48:32.736Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":11132,"end":11466},{"type":"TextQuoteSelector","exact":"It should be noted that, given features FA and FB, the probability of any particular match canbe computed exactly: P(i ↔ j) = PA\u0001B(i|d,j) ·PA\u0000B(j|d,i). Therefore, as long as rewardR factorizes over matches as R(MA↔B) = ∑(i,j)∈MA↔B r(i ↔ j), given FA and FB, we cancompute exact gradients ∇D,θM ER(MA↔B), without resorting to sampling.","prefix":"\u0000B is analogously defined by dT.","suffix":" This means that thematching ste"}]}]}
>```
>%%
>*%%PREFIX%% B is analogously defined by dT.%%HIGHLIGHT%% ==It should be noted that, given features FA and FB, the probability of any particular match canbe computed exactly: P(i ↔ j) = PAB(i|d,j) ·PA B(j|d,i). Therefore, as long as rewardR factorizes over matches as R(MA↔B) = ∑(i,j)∈MA↔B r(i ↔ j), given FA and FB, we cancompute exact gradients ∇D,θM ER(MA↔B), without resorting to sampling.== %%POSTFIX%%This means that thematching ste*
>%%LINK%%[[#^5z2da3tnwd5|show annotation]]
>%%COMMENT%%
>Since the distance matrix d is symmetric (since l2-distance is), arent P_A->B and P_B->A the same?
>%%TAGS%%
>#question
^5z2da3tnwd5



>%%
>```annotation-json
>{"created":"2023-09-29T09:53:31.762Z","text":"How does this plausible definiton work exactly?","updated":"2023-09-29T09:53:31.762Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":13177,"end":13366},{"type":"TextQuoteSelector","exact":"e declare a match plausible if depth is not available at either location, but the epipolardistance between the points is less than \u000f pixels, in which case we neither reward nor penalize it.","prefix":"their respectivereprojections. W","suffix":" Wedeclare a match incorrect in "}]}]}
>```
>%%
>*%%PREFIX%%their respectivereprojections. W%%HIGHLIGHT%% ==e declare a match plausible if depth is not available at either location, but the epipolardistance between the points is less than  pixels, in which case we neither reward nor penalize it.== %%POSTFIX%%Wedeclare a match incorrect in*
>%%LINK%%[[#^ajpdz68bxs9|show annotation]]
>%%COMMENT%%
>How does this plausible definiton work exactly?
>%%TAGS%%
>#question
^ajpdz68bxs9


>%%
>```annotation-json
>{"created":"2023-09-29T09:59:51.291Z","updated":"2023-09-29T09:59:51.291Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":14464,"end":14921},{"type":"TextQuoteSelector","exact":"It should also be noted that our formulation does not provide the feature extraction network with anysupervision other than through the quality of matches those features participate in, which means thata keypoint which is never matched is considered neutral in terms of its value. This is a very usefulproperty because keypoints may not be co-visible across two images, and should not be penalizedfor it as long as they do not create incorrect associations.","prefix":" of FA,FB with an empirical sum.","suffix":" On the other hand, this may lea"}]}]}
>```
>%%
>*%%PREFIX%%of FA,FB with an empirical sum.%%HIGHLIGHT%% ==It should also be noted that our formulation does not provide the feature extraction network with anysupervision other than through the quality of matches those features participate in, which means thata keypoint which is never matched is considered neutral in terms of its value. This is a very usefulproperty because keypoints may not be co-visible across two images, and should not be penalizedfor it as long as they do not create incorrect associations.== %%POSTFIX%%On the other hand, this may lea*
>%%LINK%%[[#^o75c5viej5c|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^o75c5viej5c


>%%
>```annotation-json
>{"created":"2023-09-29T10:00:55.134Z","text":"Helps the model to just find keypoints that it is able tomatch","updated":"2023-09-29T10:00:55.134Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":14922,"end":15259},{"type":"TextQuoteSelector","exact":"On the other hand, this may lead to manyunmatchable features on clouds and similar non-salient structures, which are unlikely to contribute tothe downstream task but increase the complexity in feature matching. We address this by imposingan additional, small penalty on each sampled keypoint λkp, which can be thought of as a regularizer","prefix":" create incorrect associations. ","suffix":".Inference. Once the models have"}]}]}
>```
>%%
>*%%PREFIX%%create incorrect associations.%%HIGHLIGHT%% ==On the other hand, this may lead to manyunmatchable features on clouds and similar non-salient structures, which are unlikely to contribute tothe downstream task but increase the complexity in feature matching. We address this by imposingan additional, small penalty on each sampled keypoint λkp, which can be thought of as a regularizer== %%POSTFIX%%.Inference. Once the models have*
>%%LINK%%[[#^u0lfk7u0wa|show annotation]]
>%%COMMENT%%
>Helps the model to just find keypoints that it is able tomatch
>%%TAGS%%
>
^u0lfk7u0wa


>%%
>```annotation-json
>{"created":"2023-09-29T10:04:23.396Z","updated":"2023-09-29T10:04:23.396Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":17248,"end":17463},{"type":"TextQuoteSelector","exact":"Although the matching stage has a single learnable parameter, θM, we found thatgradually increasing it with a fixed schedule works well, leaving just the feature extraction network tobe learned with gradient descent","prefix":"m/cvlab-epfl/disk.Optimization. ","suffix":". Since the training signal come"}]}]}
>```
>%%
>*%%PREFIX%%m/cvlab-epfl/disk.Optimization.%%HIGHLIGHT%% ==Although the matching stage has a single learnable parameter, θM, we found thatgradually increasing it with a fixed schedule works well, leaving just the feature extraction network tobe learned with gradient descent== %%POSTFIX%%. Since the training signal come*
>%%LINK%%[[#^48jc8fdw392|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^48jc8fdw392


>%%
>```annotation-json
>{"created":"2023-09-29T10:07:32.433Z","updated":"2023-09-29T10:07:32.433Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":19112,"end":19310},{"type":"TextQuoteSelector","exact":"Finally, our method produces a variable number of features. To compare it to others under a fixedfeature budget, we subsample them by their “score”, that is, the value of heatmap K at that location.","prefix":".1 and the appendix for details.","suffix":"4.1 Evaluation on the 2020 Image"}]}]}
>```
>%%
>*%%PREFIX%%.1 and the appendix for details.%%HIGHLIGHT%% ==Finally, our method produces a variable number of features. To compare it to others under a fixedfeature budget, we subsample them by their “score”, that is, the value of heatmap K at that location.== %%POSTFIX%%4.1 Evaluation on the 2020 Image*
>%%LINK%%[[#^lamom0lxh2a|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^lamom0lxh2a


>%%
>```annotation-json
>{"created":"2023-09-29T10:11:59.479Z","text":"finds a lot of matches AND, since its working patch-wise, probably uniformly distributed matches","updated":"2023-09-29T10:11:59.479Z","document":{"title":"DISK.pdf","link":[{"href":"urn:x-pdf:905b56efa88bf24b15bc549784991246"},{"href":"vault:/Sources/Image Matching/DISK.pdf"}],"documentFingerprint":"905b56efa88bf24b15bc549784991246"},"uri":"vault:/Sources/Image Matching/DISK.pdf","target":[{"source":"vault:/Sources/Image Matching/DISK.pdf","selector":[{"type":"TextPositionSelector","start":35219,"end":35304},{"type":"TextQuoteSelector","exact":"It can easily train from scratch, and yields many more matches than its competitors. ","prefix":"end to end with policy gradient.","suffix":"We demonstratestate-of-the-art r"}]}]}
>```
>%%
>*%%PREFIX%%end to end with policy gradient.%%HIGHLIGHT%% ==It can easily train from scratch, and yields many more matches than its competitors.== %%POSTFIX%%We demonstratestate-of-the-art r*
>%%LINK%%[[#^z90r2rv2wj|show annotation]]
>%%COMMENT%%
>finds a lot of matches AND, since its working patch-wise, probably uniformly distributed matches
>%%TAGS%%
>
^z90r2rv2wj
