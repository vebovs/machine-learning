import numpy as np
import tensorflow as tf
import emoji

batch_size = 1

#char_encodings = [
#    [1, 0, 0, 0, 0, 0, 0, 0],  # ' '
#    [0, 1, 0, 0, 0, 0, 0, 0],  # 'h'
#    [0, 0, 1, 0, 0, 0, 0, 0],  # 'e'
#    [0, 0, 0, 1, 0, 0, 0, 0],  # 'l'
#    [0, 0, 0, 0, 1, 0, 0, 0],  # 'o'
#    [0, 0, 0, 0, 0, 1, 0, 0],  # 'w'
#    [0, 0, 0, 0, 0, 0, 1, 0],  # 'r'
#    [0, 0, 0, 0, 0, 0, 0, 1]   # 'd'
#]
char_encodings = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]
encoding_size = np.shape(char_encodings)[1]

#index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']
index_to_char = [' ', 'c', 'a', 't']
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

#x_train = [[[char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7]]]]  # ' hello world'
#y_train = [[[char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3], char_encodings[4], char_encodings[0], char_encodings[5], char_encodings[4], char_encodings[6], char_encodings[3], char_encodings[7], char_encodings[0]]]]  # 'hello world '
x_train = [[[char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[3]]]]
y_train = [[[char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[0]]]]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape=(None, encoding_size), return_sequences=True))
model.add(tf.keras.layers.Dense(encoding_size, activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.05)

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer)

def on_epoch_end(epoch, data):
    if epoch % 10 == 9:
        print("epoch", epoch)
        print("loss", data['loss'])

        # Generate text from the initial text
        text = ' c'
        for i in range(50):
            x = np.zeros((1, i + 2, encoding_size))
            for t, char in enumerate(text):
                x[0, t, char_to_index[char]] = 1
            y = model.predict(x)[0][-1]
            text += index_to_char[y.argmax()]
        print(text.strip())
        print(emoji.emojize(':' + text.strip() + ':'))


model.fit(x_train, y_train, batch_size=batch_size, epochs=500, verbose=False, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)])

char_encodings = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]
encoding_size = np.shape(char_encodings)[1]
index_to_char = [' ', 'b', 'a', 't']
char_to_index = dict((char, i) for i, char in enumerate(index_to_char))

x_test = [[[char_encodings[0], char_encodings[1], char_encodings[2], char_encodings[3]]]]

prediction = model.predict(x_test, batch_size=batch_size, verbose=0, steps=None, callbacks=None)

result = [[[0], [0], [0], [0]]]
for i in range(len(prediction)):
    for j in range(len(prediction[i])):
        result[i][j][0] = np.where(prediction[i][j] == np.amax(prediction[i][j]))

values = []
for i in range(len(index_to_char)):
    values.append(result[0][i][0][0][0])

word = []
for i in range(len(index_to_char)):
    word.append(index_to_char[values[i]])

word = ''.join(word)
print(word.strip())
print(emoji.emojize(':' + word.strip() + ':'))