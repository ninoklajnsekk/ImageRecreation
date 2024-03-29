﻿# ImageRecreation
An application that uses a genetic algorithm to create an image that the taught classification model recognizes as the selected class.


We create the model in the ImageLearning.py file, we use the cifar10 dataset.

![image](https://user-images.githubusercontent.com/24419447/59567191-d1a9a580-906a-11e9-931b-9d1376ce62e4.png)


The fscore of the model

![image](https://user-images.githubusercontent.com/24419447/59567206-04ec3480-906b-11e9-84bc-73df560dc386.png)


Detailed fscore

![image](https://user-images.githubusercontent.com/24419447/59567218-1e8d7c00-906b-11e9-8724-333d4a7e0d7f.png)


We use the genetic algorithm to create the image, the progression looks like this:

![image](https://user-images.githubusercontent.com/24419447/59567228-36650000-906b-11e9-85d0-27d180332a34.png)


And an example result for the class_index 2 in cifar10 dataset, which represents a bird

![image](/result2.png)

with the accuracy score of: <b>0.9958367347717285
