import tensorflow as tf
import batch_builder as bb
import prepare_dataset

image_height = 300
image_width = 300

categories_count = 12

learning_rate = 1e-4
epoch_count = 1
training_batch_size = 10
validation_batch_size = 20

tf.reset_default_graph()

# Load Data
train_dataset = prepare_dataset.prepare_training_files()
test_dataset = prepare_dataset.prepare_testing_files()

train_dataset_length = len(train_dataset)
# train_dataset_length = 0.8 * train_dataset_length

training_data_set = train_dataset[0:int(train_dataset_length)-100]
print('Training Data size:', len(training_data_set))
validation_data_set = train_dataset[int(train_dataset_length)-99: len(train_dataset)-1]
print('Validation Data size:', len(validation_data_set))

train_data = bb.BatchBuilder(training_data_set, training_batch_size)
validation_data = bb.BatchBuilder(validation_data_set, validation_batch_size)

# Placeholders

# [Batch Size, Width, Height, ColorChannelSize = 3(RGB)]
input = tf.placeholder(dtype=tf.float32, shape=[None, image_width, image_height, 3])
# [Batch Size, Categories]
output = tf.placeholder(dtype=tf.float32, shape=[None, categories_count])


# Convolution Layers
# [Batch, 300, 300, 3] -> [Batch, 300, 300, 3*16]
conv1 = tf.layers.conv2d(inputs=input, filters=3*16, kernel_size=[7,7], padding="same", activation=tf.nn.relu)
# [Batch, 300, 300, 3*16] -> [Batch, 100, 100, 3*16]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=3)

# [Batch, 100, 100, 3*16] -> [Batch, 100, 100, 3*32]
conv2 = tf.layers.conv2d(inputs=pool1, filters=3*32, kernel_size=[7,7], padding="same", activation=tf.nn.relu)
# [Batch, 100, 100, 3*32] -> [Batch, 50, 50, 3*32]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# [Batch, 50, 50, 3*32 -> [Batch, 50, 50, 3*64]
conv3 = tf.layers.conv2d(inputs=pool2, filters=3*64, kernel_size=[7,7], padding="same", activation=tf.nn.relu)
# [Batch, 50, 50, 3*64] -> [Batch, 25, 25, 3*64]
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# # [Batch, 50, 50, 3*64] -> [Batch, 50, 50, 3*128]
# conv4 = tf.layers.conv2d(inputs=pool3, filters=3*128, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
# # [Batch, 50, 50, 3*128] -> [Batch, 25, 25, 3*128]
# pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

pool4_flat = tf.reshape(pool3, [-1, 25*25*3*64])

# Dense Neural Network Layers
dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4)

dense2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4)

logits = tf.layers.dense(inputs=dropout2, units=categories_count)

# Loss
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output))

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Metrics
predictions = tf.equal(tf.arg_max(logits, 1), tf.arg_max(output, 1))
accuracy = tf.reduce_mean(tf.cast(predictions, dtype=tf.float32))

# Save Model
saver = tf.train.Saver()

# Training Loop
accuracy_step_counter = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_num in range(epoch_count):

        train_data.initialize_batches()
        while train_data.data_exist():

            # Optimize
            batch_x, batch_y = train_data.get_next_batch()
            feed_dict = {input: batch_x, output: batch_y}
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)

            print('Loss=', loss_value)
            # Accuracy
            if accuracy_step_counter % 20 == 0:
                final_accuracy = 0
                count = 0
                validation_data.initialize_batches()
                while validation_data.data_exist():
                    batch_x, batch_y = validation_data.get_next_batch()
                    feed_dict = {input: batch_x, output: batch_y}
                    accu = sess.run(accuracy, feed_dict=feed_dict)
                    final_accuracy += accu
                    count += 1

                final_accuracy /= count
                print('Accuracy=', final_accuracy)

            accuracy_step_counter += 1

    # Save Model
    save_path = saver.save(sess, 'tf_model.ckpt')
    print('Model saved in path:', save_path)