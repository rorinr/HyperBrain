I have recently begun on my master's thesis journey, where I am delving into the realm of training a deep learning image matching model using medical data. To be more specific, my primary objective is to align consecutive brain scans to create a digital 3D representation from 2D brain scans. (Please note that for the scope of my thesis, I am primarily focusing on image matching, with image registration for fine-tuning being a subsequent step.) It's worth mentioning that the images I'm dealing with are highly dimensional, with dimensions reaching 140,000². To manage this, I intend to employ a patch-wise approach.

In my pursuit of knowledge in this area, I've been immersing myself in the latest research, which led me to your work as well (MatchNet). As I navigate through this process, I've encountered a few intriguing questions, and I would be immensely grateful if you could provide some insights:

1. Regarding the performance of image matching models, do you perceive a significant disparity in performance when dealing with medical images in comparison to more conventional natural images, such as those in the IMC PhotoTourism dataset?
2. My research thus far has not revealed an unsupervised approach for image matching, ie one that can identify corresponding keypoints between two images without the presence of an annotated dataset. Is it even feasible, in your opinion?
3. I've been looking at the leaderboard on paperswithcode, which showcases the performance of various models (https://paperswithcode.com/sota/image-matching-on-imc-phototourism). How much trust would you place in the reliability of this leaderboard? I was particularly surprised to observe the presence of RootSift, a model from 2012, outperforming numerous Deep Learning approaches. Could you shed some light on your perspective regarding this phenomenon?

Your insights and expertise in this field would be immensely valuable to me. Thank you for considering my questions, and I look forward to your response.

Best regards,
Robin Pierschke