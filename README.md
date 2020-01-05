# Tensorflow 2.0 : Kubrick VS Tarantino

This Python script is an experimentation of mine to classify movie screen capture, making difference between Tarantino and Kubrick's movies. A big portion of the code come from Arun Prakash's [Build your own Image classifier with Tensorflow and Keras](https://blog.francium.tech/build-your-own-image-classifier-with-tensorflow-and-keras-dc147a15e38e), adapted for Tensorflow 2.0, using the official Tensorflow's [Image classification] (https://www.tensorflow.org/tutorials/images/classification) tutorial.

to Run the script:
```
python2 train3.py
```

## Dataset

You can found my data-set [here](http://salvatore.paris/download/training-movie-director.zip), with train, validation and test folder.
Unzip it and set the right path here:
```
DATA_SET_DIR= '../../../Trainning/training-movie-director/'
```

## Conclusion

Not very conclusive. Even with a bigger data set we always have about 50% of error.
I added some drop out to the model's layers but i was worst.
I think I would havec better result with styles of movies like Horror VS Western movies. Being a Tensorflow Noob I am open to suggestion.

## Author

* **Alexandre Salvatore** - *Initial work* - [Salvatore.paris](https://http://salvatore.paris/)