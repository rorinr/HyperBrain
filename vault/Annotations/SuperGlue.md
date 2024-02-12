annotation-target::SuperGlue.pdf

>%%
>```annotation-json
>{"created":"2023-10-20T07:28:32.991Z","updated":"2023-10-20T07:28:32.991Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":8717,"end":9047},{"type":"TextQuoteSelector","exact":"Consider two images A and B, each with aset of keypoint positions p and associated visual descriptorsd – we refer to them jointly (p,d) as the local features.Positions consist of x and y image coordinates as well as adetection confidence c, pi := (x,y,c)i. Visual descriptorsdi ∈ RD can be those extracted by a CNN like SuperPoint","prefix":"ctly from the data.Formulation: ","suffix":"2Self CrossL dustbin score+Atten"}]}]}
>```
>%%
>*%%PREFIX%%ctly from the data.Formulation:%%HIGHLIGHT%% ==Consider two images A and B, each with aset of keypoint positions p and associated visual descriptorsd – we refer to them jointly (p,d) as the local features.Positions consist of x and y image coordinates as well as adetection confidence c, pi := (x,y,c)i. Visual descriptorsdi ∈ RD can be those extracted by a CNN like SuperPoint== %%POSTFIX%%2Self CrossL dustbin score+Atten*
>%%LINK%%[[#^mfox9ix34sh|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^mfox9ix34sh


>%%
>```annotation-json
>{"created":"2023-10-20T07:29:17.838Z","updated":"2023-10-20T07:29:17.838Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":8139,"end":8296},{"type":"TextQuoteSelector","exact":"a keypoint can have at most a single corre-spondence in the other image; and ii) some keypoints willbe unmatched due to occlusion and failure of the detector","prefix":"certain physicalconstraints: i) ","suffix":".An effective model for feature "}]}]}
>```
>%%
>*%%PREFIX%%certain physicalconstraints: i)%%HIGHLIGHT%% ==a keypoint can have at most a single corre-spondence in the other image; and ii) some keypoints willbe unmatched due to occlusion and failure of the detector== %%POSTFIX%%.An effective model for feature*
>%%LINK%%[[#^gnmw4m1u9rt|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^gnmw4m1u9rt


>%%
>```annotation-json
>{"created":"2023-10-20T12:54:58.255Z","text":"Previous methods introduced 'better' ways for similarity measurement, ie if 2 descriptors belong together. The here mentioned 'simple matching heuristic' is often simply euclidean distance. SuperGlue wants to learn this matching process from pre-existing local features.","updated":"2023-10-20T12:54:58.255Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":1811,"end":2044},{"type":"TextQuoteSelector","exact":"Instead of learning better task-agnostic local features followed by simple matching heuris-tics and tricks, we propose to learn the matching processfrom pre-existing local features using a novel neural archi-tecture called SuperGlue.","prefix":"ut thefeature matching problem. ","suffix":" In the context of SLAM, whichty"}]}]}
>```
>%%
>*%%PREFIX%%ut thefeature matching problem.%%HIGHLIGHT%% ==Instead of learning better task-agnostic local features followed by simple matching heuris-tics and tricks, we propose to learn the matching processfrom pre-existing local features using a novel neural archi-tecture called SuperGlue.== %%POSTFIX%%In the context of SLAM, whichty*
>%%LINK%%[[#^lt45j95bd1m|show annotation]]
>%%COMMENT%%
>Previous methods introduced 'better' ways for similarity measurement, ie if 2 descriptors belong together. The here mentioned 'simple matching heuristic' is often simply euclidean distance. SuperGlue wants to learn this matching process from pre-existing local features.
>%%TAGS%%
>
^lt45j95bd1m


>%%
>```annotation-json
>{"created":"2023-10-20T12:58:21.596Z","updated":"2023-10-20T12:58:21.596Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":3231,"end":3372},{"type":"TextQuoteSelector","exact":"ses self- (intra-image)and cross- (inter-image) attention to leverage both spatialrelationships of the keypoints and their visual appearance.","prefix":"ss of the Transformer [61], it u","suffix":"This formulation enforces the as"}]}]}
>```
>%%
>*%%PREFIX%%ss of the Transformer [61], it u%%HIGHLIGHT%% ==ses self- (intra-image)and cross- (inter-image) attention to leverage both spatialrelationships of the keypoints and their visual appearance.== %%POSTFIX%%This formulation enforces the as*
>%%LINK%%[[#^udktf8exwz|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^udktf8exwz


>%%
>```annotation-json
>{"created":"2023-10-20T13:01:13.254Z","text":"Requires another front-end model","updated":"2023-10-20T13:01:13.254Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":4554,"end":4697},{"type":"TextQuoteSelector","exact":"Whencombined with SuperPoint [18], a deep front-end, Super-Glue advances the state-of-the-art on the tasks of indoor andoutdoor pose estimation","prefix":"and learned inlier classifiers. ","suffix":" and paves the way towards end-t"}]}]}
>```
>%%
>*%%PREFIX%%and learned inlier classifiers.%%HIGHLIGHT%% ==Whencombined with SuperPoint [18], a deep front-end, Super-Glue advances the state-of-the-art on the tasks of indoor andoutdoor pose estimation== %%POSTFIX%%and paves the way towards end-t*
>%%LINK%%[[#^6l7mbntd9dr|show annotation]]
>%%COMMENT%%
>Requires another front-end model
>%%TAGS%%
>
^6l7mbntd9dr


>%%
>```annotation-json
>{"created":"2023-10-20T13:05:48.295Z","text":"assignment structure as learnable problem","updated":"2023-10-20T13:05:48.295Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":5744,"end":5844},{"type":"TextQuoteSelector","exact":"still estimated by NN search, and thus ignore the assignmentstructure and discard visual information","prefix":"hese operate on sets of matches,","suffix":". Works that learnto perform mat"}]}]}
>```
>%%
>*%%PREFIX%%hese operate on sets of matches,%%HIGHLIGHT%% ==still estimated by NN search, and thus ignore the assignmentstructure and discard visual information== %%POSTFIX%%. Works that learnto perform mat*
>%%LINK%%[[#^qqjp4i1zfpi|show annotation]]
>%%COMMENT%%
>assignment structure as learnable problem
>%%TAGS%%
>
^qqjp4i1zfpi


>%%
>```annotation-json
>{"created":"2023-10-20T13:06:26.919Z","updated":"2023-10-20T13:06:26.919Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":6003,"end":6135},{"type":"TextQuoteSelector","exact":"our learnable middle-end simulta-neously performs context aggregation, matching, and filter-ing in a single end-to-end architecture.","prefix":"e samelimitations. In contrast, ","suffix":"Graph matching problems are usua"}]}]}
>```
>%%
>*%%PREFIX%%e samelimitations. In contrast,%%HIGHLIGHT%% ==our learnable middle-end simulta-neously performs context aggregation, matching, and filter-ing in a single end-to-end architecture.== %%POSTFIX%%Graph matching problems are usua*
>%%LINK%%[[#^s9v6p0e5vc|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^s9v6p0e5vc


>%%
>```annotation-json
>{"created":"2023-10-20T13:07:11.434Z","text":"set of edges without common vertices","updated":"2023-10-20T13:07:11.434Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":6135,"end":6149},{"type":"TextQuoteSelector","exact":"Graph matching","prefix":" single end-to-end architecture.","suffix":" problems are usually formulated"}]}]}
>```
>%%
>*%%PREFIX%%single end-to-end architecture.%%HIGHLIGHT%% ==Graph matching== %%POSTFIX%%problems are usually formulated*
>%%LINK%%[[#^s8ibywtn5kb|show annotation]]
>%%COMMENT%%
>set of edges without common vertices
>%%TAGS%%
>
^s8ibywtn5kb


>%%
>```annotation-json
>{"created":"2023-10-21T08:55:48.130Z","updated":"2023-10-21T08:55:48.130Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":10425,"end":10527},{"type":"TextQuoteSelector","exact":"Our goal is to design a neural network that predicts the as-signment P from two sets of local features","prefix":"×N as:P1N ≤1M and P>1M ≤1N . (1)","suffix":".3.1. Attentional Graph Neural N"}]}]}
>```
>%%
>*%%PREFIX%%×N as:P1N ≤1M and P>1M ≤1N . (1)%%HIGHLIGHT%% ==Our goal is to design a neural network that predicts the as-signment P from two sets of local features== %%POSTFIX%%.3.1. Attentional Graph Neural N*
>%%LINK%%[[#^j5fbff0jats|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^j5fbff0jats


>%%
>```annotation-json
>{"created":"2023-10-21T08:58:07.312Z","text":"Optimize the correspondences in a global way: Integrate more information for discriminating keypoints than just keypoint+descriptor: considering spatial/visual relationship with other keypoints, which are, e.g. similar or adjacent.\nDo this for the same and for the second image!","updated":"2023-10-21T08:58:07.312Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":10565,"end":11134},{"type":"TextQuoteSelector","exact":"Besides the position of a keypoint and its visual appear-ance, integrating other contextual cues can intuitively in-crease its distinctiveness. We can for example consider itsspatial and visual relationship with other co-visible key-points, such as ones that are salient [32], self-similar [54],statistically co-occurring [73], or adjacent [58]. On theother hand, knowledge of keypoints in the second imagecan help to resolve ambiguities by comparing candidatematches or estimating the relative photometric or geomet-ric transformation from global and unambiguous cues.","prefix":"Attentional Graph Neural Network","suffix":"When asked to match a given ambi"}]}]}
>```
>%%
>*%%PREFIX%%Attentional Graph Neural Network%%HIGHLIGHT%% ==Besides the position of a keypoint and its visual appear-ance, integrating other contextual cues can intuitively in-crease its distinctiveness. We can for example consider itsspatial and visual relationship with other co-visible key-points, such as ones that are salient [32], self-similar [54],statistically co-occurring [73], or adjacent [58]. On theother hand, knowledge of keypoints in the second imagecan help to resolve ambiguities by comparing candidatematches or estimating the relative photometric or geomet-ric transformation from global and unambiguous cues.== %%POSTFIX%%When asked to match a given ambi*
>%%LINK%%[[#^sxxojtsntl9|show annotation]]
>%%COMMENT%%
>Optimize the correspondences in a global way: Integrate more information for discriminating keypoints than just keypoint+descriptor: considering spatial/visual relationship with other keypoints, which are, e.g. similar or adjacent.
>Do this for the same and for the second image!
>%%TAGS%%
>
^sxxojtsntl9


>%%
>```annotation-json
>{"created":"2023-10-21T09:01:51.498Z","updated":"2023-10-21T09:01:51.498Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":11620,"end":11707},{"type":"TextQuoteSelector","exact":"computes matchingdescriptors fi ∈ RD by letting the features communicatewith each other","prefix":"iven initial local features, it ","suffix":". As we will show, long-range fe"}]}]}
>```
>%%
>*%%PREFIX%%iven initial local features, it%%HIGHLIGHT%% ==computes matchingdescriptors fi ∈ RD by letting the features communicatewith each other== %%POSTFIX%%. As we will show, long-range fe*
>%%LINK%%[[#^f6qo6se0nyq|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^f6qo6se0nyq


>%%
>```annotation-json
>{"created":"2023-10-21T09:09:18.724Z","updated":"2023-10-21T09:09:18.724Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":12398,"end":12683},{"type":"TextQuoteSelector","exact":"The graph has two types of undirected edges – it is amultiplex graph [34, 36]. Intra-image edges, or self edges,Eself, connect keypoints i to all other keypoints within thesame image. Inter-image edges, or cross edges, Ecross, con-nect keypoints i to all keypoints in the other image. ","prefix":" the keypoints of both im-ages. ","suffix":"We usethe message passing formul"}]}]}
>```
>%%
>*%%PREFIX%%the keypoints of both im-ages.%%HIGHLIGHT%% ==The graph has two types of undirected edges – it is amultiplex graph [34, 36]. Intra-image edges, or self edges,Eself, connect keypoints i to all other keypoints within thesame image. Inter-image edges, or cross edges, Ecross, con-nect keypoints i to all keypoints in the other image.== %%POSTFIX%%We usethe message passing formul*
>%%LINK%%[[#^06emb3wx9o5j|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^06emb3wx9o5j


>%%
>```annotation-json
>{"created":"2023-10-21T09:13:21.701Z","updated":"2023-10-21T09:13:21.701Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":15095,"end":15241},{"type":"TextQuoteSelector","exact":"Our formulation provides maximum flexibility as thenetwork can learn to focus on a subset of keypoints basedon specific attributes (see Figure 4).","prefix":" with multi-head attention [61].","suffix":" SuperGlue can retrieveor attend"}]}]}
>```
>%%
>*%%PREFIX%%with multi-head attention [61].%%HIGHLIGHT%% ==Our formulation provides maximum flexibility as thenetwork can learn to focus on a subset of keypoints basedon specific attributes (see Figure 4).== %%POSTFIX%%SuperGlue can retrieveor attend*
>%%LINK%%[[#^s1a08508ia|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^s1a08508ia


>%%
>```annotation-json
>{"created":"2023-10-21T09:16:24.045Z","updated":"2023-10-21T09:16:24.045Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":15708,"end":16117},{"type":"TextQuoteSelector","exact":"The second major block of SuperGlue (see Figure 3) isthe optimal matching layer, which produces a partial assign-ment matrix. As in the standard graph matching formu-lation, the assignment P can be obtained by computing ascore matrix S ∈RM×N for all possible matches and max-imizing the total score ∑i,j Si,j Pi,j under the constraintsin Equation 1. This is equivalent to solving a linear assign-ment problem.","prefix":"in B.3.2. Optimal matching layer","suffix":"Score Prediction: Building a sep"}]}]}
>```
>%%
>*%%PREFIX%%in B.3.2. Optimal matching layer%%HIGHLIGHT%% ==The second major block of SuperGlue (see Figure 3) isthe optimal matching layer, which produces a partial assign-ment matrix. As in the standard graph matching formu-lation, the assignment P can be obtained by computing ascore matrix S ∈RM×N for all possible matches and max-imizing the total score ∑i,j Si,j Pi,j under the constraintsin Equation 1. This is equivalent to solving a linear assign-ment problem.== %%POSTFIX%%Score Prediction: Building a sep*
>%%LINK%%[[#^m09sk21uvkf|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^m09sk21uvkf


>%%
>```annotation-json
>{"created":"2023-10-21T09:17:26.768Z","updated":"2023-10-21T09:17:26.768Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":16223,"end":16375},{"type":"TextQuoteSelector","exact":"e in-stead express the pairwise score as the similarity of match-ing descriptors:Si,j =< fAi ,fBj >, ∀(i,j) ∈A×B, (7)where < ·,· > is the inner product.","prefix":" matches would be prohibitive. W","suffix":" As opposed to learnedvisual des"}]}]}
>```
>%%
>*%%PREFIX%%matches would be prohibitive. W%%HIGHLIGHT%% ==e in-stead express the pairwise score as the similarity of match-ing descriptors:Si,j =< fAi ,fBj >, ∀(i,j) ∈A×B, (7)where < ·,· > is the inner product.== %%POSTFIX%%As opposed to learnedvisual des*
>%%LINK%%[[#^qgc78oznceo|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^qgc78oznceo


>%%
>```annotation-json
>{"created":"2023-10-21T09:18:18.383Z","updated":"2023-10-21T09:18:18.383Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":16588,"end":16723},{"type":"TextQuoteSelector","exact":"To let the network suppresssome keypoints, we augment each set with a dustbin so thatunmatched keypoints are explicitly assigned to it.","prefix":"dence.Occlusion and Visibility: ","suffix":" This tech-nique is common in gr"}]}]}
>```
>%%
>*%%PREFIX%%dence.Occlusion and Visibility:%%HIGHLIGHT%% ==To let the network suppresssome keypoints, we augment each set with a dustbin so thatunmatched keypoints are explicitly assigned to it.== %%POSTFIX%%This tech-nique is common in gr*
>%%LINK%%[[#^eclpdz2jx8b|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^eclpdz2jx8b


>%%
>```annotation-json
>{"created":"2023-10-21T09:24:36.151Z","text":"stochastic matrix s.t. we are able to interpret values as percentages ","updated":"2023-10-21T09:24:36.151Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":17788,"end":17999},{"type":"TextQuoteSelector","exact":"It is a differentiableversion of the Hungarian algorithm [35], classically usedfor bipartite matching, that consists in iteratively normal-izing exp( ̄S) along rows and columns, similar to row andcolumn Softmax.","prefix":"he Sinkhorn algorithm [55, 12]. ","suffix":" After T iterations, we drop the"}]}]}
>```
>%%
>*%%PREFIX%%he Sinkhorn algorithm [55, 12].%%HIGHLIGHT%% ==It is a differentiableversion of the Hungarian algorithm [35], classically usedfor bipartite matching, that consists in iteratively normal-izing exp( ̄S) along rows and columns, similar to row andcolumn Softmax.== %%POSTFIX%%After T iterations, we drop the*
>%%LINK%%[[#^nkepx8zztm|show annotation]]
>%%COMMENT%%
>stochastic matrix s.t. we are able to interpret values as percentages 
>%%TAGS%%
>
^nkepx8zztm


>%%
>```annotation-json
>{"created":"2023-10-21T09:27:30.074Z","text":"Train with a standard classification loss, ie negative log likelihood in a supervised manner. Extracting ground truth matches from depth maps. This also allows to classify keypoints as unmatched if they do not have any reprojection","updated":"2023-10-21T09:27:30.074Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":18552,"end":18795},{"type":"TextQuoteSelector","exact":"iven these labels, we minimizethe negative log-likelihood of the assignment  ̄P:Loss = − ∑(i,j)∈Mlog  ̄Pi,j−∑i∈Ilog  ̄Pi,N+1 −∑j∈Jlog  ̄PM+1,j .(10)This supervision aims at simultaneously maximizing theprecision and the recall of the matching.","prefix":"-projection in their vicinity. G","suffix":"3.4. Comparisons to related work"}]}]}
>```
>%%
>*%%PREFIX%%-projection in their vicinity. G%%HIGHLIGHT%% ==iven these labels, we minimizethe negative log-likelihood of the assignment  ̄P:Loss = − ∑(i,j)∈Mlog  ̄Pi,j−∑i∈Ilog  ̄Pi,N+1 −∑j∈Jlog  ̄PM+1,j .(10)This supervision aims at simultaneously maximizing theprecision and the recall of the matching.== %%POSTFIX%%3.4. Comparisons to related work*
>%%LINK%%[[#^qs2zz62x9bt|show annotation]]
>%%COMMENT%%
>Train with a standard classification loss, ie negative log likelihood in a supervised manner. Extracting ground truth matches from depth maps. This also allows to classify keypoints as unmatched if they do not have any reprojection
>%%TAGS%%
>
^qs2zz62x9bt


>%%
>```annotation-json
>{"created":"2023-10-21T09:33:01.412Z","updated":"2023-10-21T09:33:01.412Z","document":{"title":"SuperGlue.pdf","link":[{"href":"urn:x-pdf:d0697fa0ce329834ea77a8ef9c8cd232"},{"href":"vault:/Sources/Image Matching/SuperGlue.pdf"}],"documentFingerprint":"d0697fa0ce329834ea77a8ef9c8cd232"},"uri":"vault:/Sources/Image Matching/SuperGlue.pdf","target":[{"source":"vault:/Sources/Image Matching/SuperGlue.pdf","selector":[{"type":"TextPositionSelector","start":20616,"end":20682},{"type":"TextQuoteSelector","exact":"do not train the visual descriptor network whentraining SuperGlue.","prefix":"nless explicitly men-tioned, we ","suffix":" At test time, one can use a con"}]}]}
>```
>%%
>*%%PREFIX%%nless explicitly men-tioned, we%%HIGHLIGHT%% ==do not train the visual descriptor network whentraining SuperGlue.== %%POSTFIX%%At test time, one can use a con*
>%%LINK%%[[#^vuwsu1ohfol|show annotation]]
>%%COMMENT%%
>
>%%TAGS%%
>
^vuwsu1ohfol
