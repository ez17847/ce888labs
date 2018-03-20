# Lab8

## Values obtained

###Original inputs:
Test score: 0.6869393751072883
Test accuracy: 0.8366

###First modification
inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = Flatten()(x)
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)

Test score: 1.1115880798298121
Test accuracy: 0.82184

###Second modification
inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)
x = Dense(64)(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)

Test score: 1.1945538347268105
Test accuracy: 0.81008


###Third modification
inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)
x = Convolution1D(filters = 32 ,kernel_size = 10, activation = 'relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)

Test score: 1.010499802054167
Test accuracy: 0.82696

###Fourth modification
inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)
x = LSTM(32)(x)
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)

Test score: 1.215005755226463
Test accuracy: 0.81416

