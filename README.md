<img src='imgs/input_00001_1_0.64_0.07_0.83_0.14_0.20.jpg' align="right" width=384>
The dataset consists of a series of images like the one shown above. Some examples are provided in the dataset folder. For more data, please contact the author.

Train a model by running train.py 
To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/result/web/index.html`.
A trained model is provided in ./checkpoints/result,named latest_net_G_A.pth and latest_net_G_B.pth.

Test the model:
Test the full net by running test_full_net.py on the data from ./dataset/test
Test the normal reconstruction net by running test_G_A.py on the real water images from ./dataset/test_real
<img src='imgs/test_result_00298_1_0.18_0.20_0.97_0.60_0.25.jpg' align="right" width=384>
<img src='imgs/test-G_A_result_00026_5_0.62_0.05_0.42_0.25_0.86.jpg' align="right" width=384>

<br><br><br>



