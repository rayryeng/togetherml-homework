# Homework 3

**This assignment will require research into the FastAI library, posts on Discord or Slack that eliminate the need of people to do this research will be deleted**

## Assignment Details

* Watch the lesson 3 video at course.fastai.com
* Start a new notebook
* Using the CamVid (`URLs.CAMVID`) from the FastAI datasets) dataset train a UNet to at least 93% accuracy
* Once your model is trained, export it
* Take 3 of **your own pics** (as in with your cell phone, not from the internet) at street view of scenes that may contain roads, cars, people, buildings, trees, grass, bushes, etc (in any combination)
* Load the model from the export.
* Using each of the pics you took, create a prediction using the model you loaded.
* Show the pic, with the prediction overlaied on top of the pic.  the figure size should be (12,12).
* You may do this in 3 separate cells if you wish

After your predictions, answer the following questions in a separate markdown cell:
* How good was your model at predicting this "holdout" test set?  Where is it having difficulty or particular success?
* Describe 2 strategies that **you** could legitimately use, with the resources you have, to increase this quality given that you would not be privy to these images.

## Ray's Edits

I trained using the `fastai` distributed paradigm.  Unfortunately, this means
that this cannot be run in a notebook environment.  The training portion
of this assignment was placed in `week3_train.py`.  Once the final model
was created, we use the notebook to load in the exported model then load in
our actual images for testing and visualize what they look like overlaid
with the predictions.

Please note that the final model is larger than 100 MB which is Github's
limit for file sizes before having to invest in LFS so the model is hosted
offline on my personal Dropbox.  Please access it here and place it
in the `models` subdirectory.

https://www.dropbox.com/s/3y87o49q35x171o/camvid-final.pkl?dl=1