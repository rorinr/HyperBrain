- The paper didnt mention the masking which is part of the implementation in github. How important is this masking step during training? Did you even used it?
	- What are the exact differences between train vs test-time? Just the masking?
- There is a focal loss and a plain dual softmax implemented. The focal loss is not mentioned in the paper. What do they recommened?
- Explain idea of superbrain and ask for their opinion if its possible.
- There is also a masking at test time - why?
- In the coarse matching step the last update seems off: Why checking for ==0 and != 0?




## More questions
- what does sparse supervision mean?
- why no dustbin for dual softmax?