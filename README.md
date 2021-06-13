## MODEL 1 : 3 Layers with 1 Convolution layer
```python
    if model_number == 1:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2 
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 3
```

### Test loss :  0.1132686400660372
### Test accuracy : 0.9746

### Images and corresponding probability that predicted Right 
![1_1](https://user-images.githubusercontent.com/37043329/121804940-ecc44580-cc83-11eb-8869-769721901e26.JPG)
### Images and corresponding probability that predicted Wrong
![1_2](https://user-images.githubusercontent.com/37043329/121804959-ffd71580-cc83-11eb-82f4-5408d2c78e1b.JPG)

## MODEL 2 : 5 Layers with 2 Convolution layer
```python
    if model_number == 2:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),     # layer 1 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 4
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 5
```

### Test loss :  0.04249909892678261
### Test accuracy : 0.9871

### Images and corresponding probability that predicted Right 
![2_1](https://user-images.githubusercontent.com/37043329/121805118-b76c2780-cc84-11eb-8989-16e9327daadd.JPG)

### Images and corresponding probability that predicted Wrong
![2_2](https://user-images.githubusercontent.com/37043329/121805121-bb984500-cc84-11eb-83ad-247f1a9ffe3d.JPG)

## MODEL 3 : 7 Layers with 4 Convolution layer
```python
    if model_number == 3: 
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 4
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 5
                    keras.layers.Conv2D(128, (3,3), activation = 'relu'),                           # layer 6
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 7
```

### Test loss :  0.03976685698558358
### Test accuracy : 0.9884

### Images and corresponding probability that predicted Right 
![3_1](https://user-images.githubusercontent.com/37043329/121805282-86402700-cc85-11eb-8dee-47c9f45a15e0.JPG)

### Images and corresponding probability that predicted Wrong
![3_2](https://user-images.githubusercontent.com/37043329/121805287-8c360800-cc85-11eb-9748-ed8cbeb20083.JPG)



