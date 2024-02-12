- explain the difference between train on aligned images vs original images: We cant train directly on aligned images because the model would just return the same pixel positions
- Explain the problem of using rotation. Is there any solution to this?
- Positional encoding useful when we dont work with the whole image and dont have access to the original images?
- sampling of crops -> matching area probably not uniformly distributed wrt the original image?!
- focal loss vs dual softmax
- dual softmax has no dustbin -> problem?


-> use rotation and cropping/shifting
use affine transformation

-> maybe allow one row to match multiple points

-> maybe mehrere paare erlauben 

-> bicubic sampling