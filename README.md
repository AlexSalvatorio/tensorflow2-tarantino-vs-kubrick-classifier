
## 2020/01/11 Update
**There was a major error in my previous code, concerning the Test display, giving a wrong idea of the model predictions. I corrected the bug and added shuffling to the test sample.**

# Tensorflow 2.0 : Kubrick VS Tarantino

This Python script is an experimentation of mine classifying screen captures from directors Tarantino and Kubrick. A big portion of the code come from Arun Prakash's [Build your own Image classifier with Tensorflow and Keras](https://blog.francium.tech/build-your-own-image-classifier-with-tensorflow-and-keras-dc147a15e38e), adapted for Tensorflow 2.0, and also the official Tensorflow's [Image classification] (https://www.tensorflow.org/tutorials/images/classification) tutorial.

to Run the script:
```
python2 train3.py
```

## Dataset

You can found my data-set [here](http://salvatore.paris/download/training-movie-director.zip), with /train, /validation and /test folders.
Unzip it and set the right path here:
```
DATA_SET_DIR= '../../../Trainning/training-movie-director/'
```

## Conclusion

Not very conclusive.
~~Even with a bigger data set we always have about 50% of error.
I added some drop out to the model's layers but it was worst.
I think I will have better result with movie genre classification (like Horror VS Western). Being a Tensorflow Noob I am open to suggestion to improve it.~~
**2020/01/11 Update**
To be honest i don't think this type of classification is relevant. Some screenshoot in kind of composition/shade/themes are very similar between the two directors. Note than adding a 1:1.29 ratio to the size make a significant improvement in the performance of the model. 

Learning graphs here:
![Image of loss and accuracy](https://pbs.twimg.com/media/ENhyJkWXUAIlwJP?format=jpg)
Results with the /test files folder (Note than _Once Upon a Time in Hollywood_ wasn't in the /train nor /validation folders)
![Image of final model test]( https://pbs.twimg.com/media/ENhyJkcWsAAVmbd?format=jpg)

## Author

* **Alexandre Salvatore** - *Initial work* - [Salvatore.paris](https://http://salvatore.paris/)