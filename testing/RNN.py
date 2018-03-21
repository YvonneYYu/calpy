import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Reshape, Lambda
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras import backend as K


data = np.load('data.npy').item()
X_train, Y_train, X_test, Y_test = data['X_train'], data['Y_train'], data['X_test'], data['Y_test']
#X.shape = (m, T_x, n_x). m is the number of examples, T_x is the time length, n_x is the feature dimension
#Y.shape = (T_y, m, n_x). Y is essentially the same as X, just shifted one step to the left. T_y == T_x herein.
m, T_x, n_x = X_train.shape
n_a = 64 #number of hidden states

reshapor = Reshape((1, n_x))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_x, activation='sigmoid')

def trouble_model(T_x, n_a, n_x):
    """
    Implement the model
    
    Arguments:
    T_x -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_x -- number of unique pause patterns 
    
    Returns:
    model -- a keras model 
    """
    
    # Define the input of model with a shape 
    X = Input(shape=(T_x, n_x))
    
    # Define a0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []
    
    for t in range(T_x):
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshapor(x)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        
    model = Model(inputs=[X,a0, c0], outputs=outputs)    
    return model

model = trouble_model(T_x = T_x , n_a = n_a, n_x = n_x)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

#define f1 score as metrics
def f1(y_true, y_pred):

    true_p = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_p = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_p = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    recall = true_p / (possible_p + K.epsilon())
    precision = true_p / (predicted_p + K.epsilon())

    return 2 * precision * recall / (precision + recall)


model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[f1, binary_accuracy])

#zero initialisation 
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))


model.fit([X_train, a0, c0], list(Y_train), epochs=100)
Y_pred = model.predict([X_test, a0, c0])
score = model.evaluate([X_test, a0, c0], list(Y_test))