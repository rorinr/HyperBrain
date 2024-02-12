- Developed for image retrieval
- Train a weak-supervised CNN classifier (weak = without the need of object- and patch-level annotations. Also the labels are noisy through GPS-noise) ([[Annotations/DELF#^iojkpsde52q]])
- At test-time, we get descriptors and keypoints via one forward pass ([[Annotations/DELF#^iojkpsde52q]])
- The Google-landmarks dataset:  Images associated with geolocations. They used the geolocation as label for the image, s.t. two images belong to the same landmark if distance is less than a threshold (25km) ([[Annotations/DELF#^0afpxe4ryj8]])
- Train ResNet50 with an image pyramid (to handle scale changes) and obtain the keypoint/feature location = the pixel coordinates of the center of the receptive field ([[Annotations/DELF#^maci466l42]])
- They claim that through the randomcrop, local descriptors implicitly learn representations that are more relevant for the landmark retrieval problem ([[Annotations/DELF#^tzi3pnezgbo]])
- The now extracted features are partially irrelevant. The next step is to train a keypoint selection ([[Annotations/DELF#^jrwcoctgpuq]])
- There are N different d-dimensional feature-vectors. We sum these N vectors up to one single vector, but in a weighted way (these vectors are weighted by alpha). The resulting sum, a single d-dimensional vector, gets multiplied with W for the final prediction, which is classification again ([[Annotations/DELF#^tb9iebba6do]])
- Even if possible, learning descriptors and then the score function alpha yields in better results ([[Annotations/DELF#^ugking3s9jf]])
- Describe-than-keypoint ([[Annotations/DELF#^9hypjgnjxjb]])
- They used Nearest neighbor search for finding corresponding images ([[Annotations/DELF#^e78cl8ao4bn]])

### DELF questions
- attention architecture not further explained
- How did they decide which features to use? Just the n features with highest attention-score or a threshold?
