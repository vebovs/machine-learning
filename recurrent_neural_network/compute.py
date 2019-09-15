import numpy as np
import tensorflow as tf
import emoji

batch_size = 100

char_encodings = [
    [1, 0, 0, 0, 0, 0], #c
    [0, 1, 0, 0, 0, 0], #a
    [0, 0, 1, 0, 0, 0], #t
    [0, 0, 0, 1, 0, 0], #r
    [0, 0, 0, 0, 1, 0], #s
    [0, 0, 0, 0, 0, 1]  #h
]
encoding_size = np.shape(char_encodings)[1]

index_to_char = ['c', 'a', 't', 'r', 's', 'h']
categories = ['cat', 'rat', 'hat', '', '', '']
index_categories = [
    [1, 0, 0, 0, 0, 0], #cat
    [0, 1, 0, 0, 0, 0], #rat
    [0, 0, 1, 0, 0, 0], #hat
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

x_train = [[[char_encodings[0], char_encodings[1], char_encodings[2]], [char_encodings[3], char_encodings[1], char_encodings[2]], [char_encodings[5], char_encodings[1], char_encodings[2]]]] #cat, rat, hat
y_train = [[[index_categories[0]], [index_categories[1]], [index_categories[2]]]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, encoding_size), return_sequences=True))
model.add(tf.keras.layers.Dense(encoding_size, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.05)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer)

print('Write a word to be learned and converted into an emoji (enter blank to exit the application)')
text = input('Word: ')
while(text):
    def on_epoch_end(epoch, data):
        if epoch % 10 == 9:
            print("epoch", epoch)
            print("loss", data['loss'])
            global text
            for i in range(50):
                x = np.zeros((1, i + len(text) + 1, encoding_size))
                for t, char in enumerate(text):
                    x[0, t, char_to_index[char]] = 1
                y = model.predict(x)[0][-1]
                text = categories[y.argmax()]
            print(emoji.emojize(':' + text + ':'))


    model.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=False, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])
    print('Write a word to be learned and converted into an emoji (enter blank to exit the application)')
    text = input('Word: ')