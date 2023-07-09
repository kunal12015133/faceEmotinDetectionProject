# FER2013 

Kaggle Challenge - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Facial Emotion Recognition on FER2013 Dataset Using a Convolutional Neural Network. 


This Model -  66.369% accuracy


These instructions will get this model up and running. Follow them to make use of the `fertestcusstom.py` file to recognize facial emotions using custom images. This model can also be used as facial emotion recognition part of projects with broader applications

### Prerequisites
Install these prerequisites before proceeding-
```
 pip3 install tensorflow
 pip3 install keras
 pip3 install numpy
 pip3 install sklearn
 pip3 install pandas
 pip3 install opencv-python
```

### Method 1 : Using the built model 

If you don't want to train the classifier from scratch, you can make the use of `fertestcustom.py` directly as the the repository already has `fer.json` (trained model) and `fer.h5` (parameters) which can be used to predict emotion on any test image present in the folder. You can modify `fertestcustom.py` according to your requirements and use it to predict fatial emotion in any use case.

### Method 2 : Build from scratch
Downlaod the repo or clone it
```

```
Download and extract the dataset from Kaggle link above.

Run the `preprocessing.py` file, which would generate `fadataX.npy` and `flabels.npy` files for you.

Run the `fertrain.py` file,  this would take sometime depending on your processor and gpu. Took around 1.5 hour for with an Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz   2.30 GHz. This would create `modXtest.npy`, `modytest,npy`, `fer.json` and `fer.h5` file for you.

## Running the tests (Optional)

You can test the accuracy of trained classifier using `modXtest.npy` and `modytest.npy` by running `fertest.py` file. This would give youy the accuracy in % of the recently trained classifier.

## Getting the Confusion Matrix (Optional)

You can get the confusion matrix for this model by running `confmatrix.py` file. This would evaluate the most confused expressions and generate a `confusionmatrix.png` file for your trained model.


# Model Summary

The layers in the Convolution Neural Network used in implementing this classifier can be summarized as follows. You can git a similar summary by decommenting the `model.summar()` function before executing `fertrain.py` file.

#Real Time Testing

Open main.py using python to run test


