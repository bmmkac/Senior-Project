我晚上有事要出去，先发你


Detect.py 是给的trainer 和detect的合并版，改了超参数和optimizer

box_to_im 现在用不到

Cnn  这个是用来auto formating 的模型7层lenet5，逻辑是用detector 把图里面的bounding box 找出来，然后把bounding box 扔到 box_to_im 里面出图片，再扔到cnn里来train或者test
