*Originally posted by mmuckley in October of 2020*

*Original URL: https://discuss.fastmri.org/t/2020-fastmri-reconstruction-challenge-launch/208*

We’re excited to announce that the fastMRI challenge data has been released and the challenge has begun. As with last year, to make a submission, go to the “Submit Reconstruction” page on fastmri.org 43 and make sure you check the “Challenge” submission type. Submissions can be made until October 15th at 11:59 UTC.

As a reminder, this year’s challenge features three tracks: Muli-Coil 4X, Multi-Coil 8X, and Transfer. The Multi-Coil 4X and Multi-Coil 8X tracks are both on Siemens neurological imaging data, whereas the Transfer track contains a combination of GE and Philips neurological data. The datasets can be downloaded from the [fastMRI NYU dataset page](https://fastmri.med.nyu.edu/). Just fill out your details and a link to the multicoil_brain_challenge.tar.gz (containing 4X and 8X Siemens data) and multicoil_brain_challenge_transfer.tar.gz (containing 4X GE and Philips data) files will be sent to you.

For the data in the “Transfer” track, we’d like to note a caveat that the GE data is provided without frequency oversampling. The lack of frequency oversampling is due to the analog-to-digital conversion process on the GE scanners. This might cause issues for models that were trained with frequency oversampling, such as those in our own GitHub repository. To illustrate the problem, we have added a [new folder to the GitHub repository](https://github.com/facebookresearch/fastMRI/tree/master/experimental/brain_challenge_inference) with inference scripts for all three tracks of the challenge. In our Transfer inference, we simulate frequency oversampling with zero-padding. We believe there are better ways to augment the Transfer data with regards to frequency oversampling. Our only goal with providing this code is to illustrate the issue to challenge participants. For further details, please see the README for the inference scripts.

Please see this [blog post](https://ai.facebook.com/blog/the-2020-fastmri-challenge-opens-for-submissions-on-october-1/) and the [fastmri.org](http://fastmri.org/) website for further information. You can contact us through this message board or email us at fastmri@fb.com if you have any questions.

Good luck and kind regards,
The fastMRI team
