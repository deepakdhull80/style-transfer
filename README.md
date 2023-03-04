# style-transfer
Neural style transfer - A Neural Algorithm of Artistic Style

## Setup:
```bash
conda create -n env python==3.9
conda activate env
git clone https://github.com/deepakdhull80/style-transfer.git
```
## inference
```bash
python run.py -ci content.jpg -si style.jpg

[Optional args]
  --device -d : [default|cpu]
  --iteration -it : [default|200]
  --image_size : [default|512]
```

## Some results

#### content and style
<p align="center">
  <img src="https://github.com/deepakdhull80/style-transfer/blob/main/images/content.jpg" width="256" height="256" title="content image">
  <img src="https://github.com/deepakdhull80/style-transfer/blob/main/images/style.jpg" width="256" height="256" title="style image">
  <br><br>
  <img src="https://github.com/deepakdhull80/style-transfer/blob/main/images/result.jpg" width="340" height="340" title="result image">
</p>


### Learning
1. LBFGS optimizer
This method of optimization, where we take into account the objective function’s second order behavior in addition to its first order behavior, is known as Newton’s method.

