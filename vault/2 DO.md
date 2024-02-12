 
- [ ] Find out whiy refinmenet model decreasees performance
- [ ] PIL.Image.LANCZOS used in read_image. WHy?
- [ ] How is the connection between coarse and fine concretly implemented in loftr, especially for training?
- [ ] edge case if no matches found -> Still important to have for test time!
- [ ] implement some form of early stopping here
- [ ] write evaluation function that tests a trained model on n samples of the test dataset. This function should return multiple performance metrics: accuracy coarse matcher, average offset in pixel and the metric used in paper
	- [ ] save this for: just coarse matching, coarse + refinement matching
- [ ] refactor the whole code base
	- [ ] write obsidian documentation for each part (maybe also using images eg for backbone)
- [ ] the one to many condition of match matrix could yield problems in refinement step
- [ ] test trained model on the original data
- [ ] rename i_ids and j_ids to matches_0 and matches_1 or so
- [ ] use the masking for borders thats also in original loftr to avoid the need of padding in fine-matching (when choosing the windows)
- [ ] changed in the fine loftr transformer d_model =92 and n_head=4. Is that legit?
- [ ] baseline: Loftr coarse- check if fine-step increases performance
- [ ] it says at one step in coarse matching that this only works if at most one True per row. Does this affect my approach?
- [ ] make sure that transformation cant yield in inability to sample crop_start
- [ ] Refactor the code where needed: torch.non_zero, no masking etc. -> Make it simple
- [ ] go more into lightning and hydra
- [ ] verify loss @ init: see karpathy link to details
- [ ] Init the final layers (i think it is just relevant for the final coarse matching layer) with the correct bias like in the deep learning recipe from karpathy 
- [ ] input-indepent baseline - zeros as input
- [ ] overfit one batch - every single prediction should be correct here to reach minimal loss. see again the recipe
- [ ] Visualize prediction dynamics on fixed test batch during training
- [ ] use backprop to chart dependencies see the recipe for further instructions
	1. Create a multi-batch input (x = torch.rand([4, 3, 224, 224])  
	2. Set your input to be differentiable (x.requires_grad = True)  
	3. Run a forward pass (out = model(x))  
	4. Define the loss as depending from only one of the inputs (for instance: loss = out[2].sum())  
	5. Run a backprop (loss.backward)  
	6. Verify that only x[2] has non-null gradients: assert (x.grad[i] == 0.).all() for i != 2 and (x.grad[2] != 0).any()
- [ ] Train a simple model in a way for a baseline like mentioned here [https://karpathy.github.io/2019/04/25/recipe/]
	- [ ] One layer cnn to 1/8 or 1/16 then linear layer to coarse match and so on. Make sure to use fixed seed for reproducing (also the used augmentations should be reproducable)
- [ ] investigate how to process all image data at once (first and second crops) in backbone (see loftr code here)
- [ ] reinvestigating the backbone, make more comments and draw pic of it for understanding. Also make research regarding feature pyramid networks




## Less urgent

- [ ] Summarize CasMTR
- [ ] Write attention comparison
	- [ ] explanation for each approach
	- [ ] table with complexity?
- [ ] Read and summarize remaining papers
- [ ] Write down Evaluation metrics
- [ ] Finish methodology



# Random stuff
- use kornia for first attempts https://kornia.readthedocs.io/en/latest/applications/image_matching.html if its possible for microscopic data
- RootSIFT! its performing really good on paperswithcode even its from 2012, use this as a baseline which doesnt require training
- https://www.kaggle.com/competitions/image-matching-challenge-2022/leaderboard
- https://www.kaggle.com/competitions/image-matching-challenge-2023/leaderboard