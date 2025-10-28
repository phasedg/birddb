Repo for FGIR on CUB_200 and nabirds

10/2025
- Revisit 2021 project and attempt to get to 2021 baseline.
- notebooks/expt_anal.ipynb documents results

Lessons learned
- correct transform pipeline
--  read image to uint8, resize, move to GPU
-- run image transforms: HFlip, randomchoice (jitter,rotate,contrast)
-- toDtype float, rescale , normalize
- for R50 models, 20 to 40 epochs suffice
- need tp explore in expt2
-- retrain all weights
-- lr rate
-- model type



