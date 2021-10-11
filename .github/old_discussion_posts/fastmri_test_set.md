*Originally Posted by jzb in June of 2019*

*Original URL: https://discuss.fastmri.org/t/issue-with-the-fastmri-test-dataset/29/*

We found an issue with the k-space masking code that affected the fastMRI test dataset. Instead of setting the masked k-space values to 0, they were set to either +0 or -0 (the IEEE 754 standard for floating-point arithmetic can represent both +0 and -0), depending on the sign of the value being masked. The sign provides extra bits of information that are not available in actual under-sampled measurements. You should have received an email from fastMRI@med.nyu.edu yesterday with instructions on how to download the new test dataset.

The issue occurred because multiplying a negative floating point number with zero produces -0.0 instead of 0.0, as demonstrated in the following code snippet:

```python
>>> 1. * 0
0.0
>>> -1. * 0
-0.0
```
We masked k-space by multiplying it with the mask:

```python
>>> import numpy as np
>>> kspace = np.array([1. + 1.j, 1. - 1.j, -1. + 1.j, -1. - 1.j])
>>> mask = np.array([0., 0., 0., 0.])
>>> kspace * mask
array([ 0.+0.j,  0.+0.j, -0.+0.j,  0.-0.j])  # notice that some zeroes have a negative sign
```

A correct masking function should produce zeros with a positive sign for all masked values. This can be achieved, for example, by using numpy’s where function:

```python
>>> np.where(mask, kspace, 0)
array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])  # all zeros have a positive sign
```

Since this is an easy mistake to make, we encourage all of you to verify that your masking code is correct. Please note that the mistake was also present in our fastMRI github repo and that we [fixed it](https://github.com/facebookresearch/fastMRI/pull/14) only recently.

The results of the baseline models from [our paper](https://arxiv.org/abs/1811.08839) are not affected by this issue, since the models we used are not able to take advantage of the additional sign information.
