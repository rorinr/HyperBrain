- Find date with Timo Dickscheid
	- 27.11 14 Uhr im Anschluss?
- How to match this extreme large images?
	- Grobe Ausrichtung der originalen Bilder so dass davon ausgegangen werden kann dass die Bilder grob Übereinstimmen (anstatt jeden Patch zu matchen)
- Clustering in feature space for unsupervised image matching?
- How could we use even superpoint/superglue/lightglue with its augmentations for this problem? Learn the augmentations? Is there a way for doing this unsupervised (unfortunately no data yet)?
	- Maybe if matching image_1 with augmented image_1, maybe this is enough to match image_1 with image_2
	- keypoint detector for micrsocopic images
- Sota [[Attention mechansims]] ?
- Go through kaggle leaderboard https://www.kaggle.com/competitions/image-matching-challenge-2023/leaderboard
- regarding [[Annotations/LightGlue#^l5q3vruge1]]: Is loftr fast enough for this problem?
	- probably no problem
- sota positional encoding?
- augmentation needed: otherwise the model will learn to just predict the same pixel-coordinates (since images are alligned)
- NMS?
- Wie Unterschied zwischen originalen daten und annotiert bzgl. der Ausrichtung? Hat das Lab schon eine Methode dieser groben Ausrichtung?
- Auf wie viel Pixel ist das matching genau?
- Originalgröße?
- Entwicklung auf GPUs auch auf Großcomputer?