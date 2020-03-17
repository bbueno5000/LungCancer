"""
DOCSTRING
"""
# standard
import math
import os
# non-standard
import cv2
import pydicom
import matplotlib.pyplot as pyplot
import numpy
import pandas
import tensorflow

class ConvNet:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        self.classes = 2
        self.data_path = 'C:\\Users\\bbuen\\Downloads\\DSB3\\stage1'
        self.keep_rate = 0.8
        self.labels = pandas.read_csv(
            'C:\\Users\\bbuen\\Downloads\\DSB3\\stage1_labels.csv', index_col=0)
        self.slice_count = 20
        self.image_size = 150 # pixels
        self.much_data = numpy.load('muchdata-50-50-20.npy')
        self.patients = os.listdir(self.data_path)
        self.train_data = self.much_data[:-100]
        self.validation_data = self.much_data[-100:]
        self.variable_x = tensorflow.placeholder('float')
        self.variable_y = tensorflow.placeholder('float')

    def __call__(self):
        """
        DOCSTRING
        """
        self.train_neural_network(self.variable_x)

    def chunks(self, list_a, number):
        """
        Yield successive n-sized chunks from l.
        """
        for i in range(0, len(list_a), number):
            yield list_a[i:i + number]

    def conv_3d(self, variable_x, variable_w):
        """
        DOCSTRING
        """
        return tensorflow.nn.conv3d(
            variable_x, variable_w, strides=[1, 1, 1, 1, 1], padding='SAME')

    def convolutional_neural_network(self, variable_x):
        """
        DOCSTRING
        """
        weights = {'W_conv1':tensorflow.Variable(tensorflow.random_normal([3, 3, 3, 1, 32])),
                   'W_conv2':tensorflow.Variable(tensorflow.random_normal([3, 3, 3, 32, 64])),
                   'W_fc':tensorflow.Variable(tensorflow.random_normal([54080, 1024])),
                   'out':tensorflow.Variable(tensorflow.random_normal([1024, self.classes]))}
        biases = {'b_conv1':tensorflow.Variable(tensorflow.random_normal([32])),
                  'b_conv2':tensorflow.Variable(tensorflow.random_normal([64])),
                  'b_fc':tensorflow.Variable(tensorflow.random_normal([1024])),
                  'out':tensorflow.Variable(tensorflow.random_normal([self.classes]))}
        variable_x = tensorflow.reshape(
            variable_x, shape=[-1, self.image_size, self.image_size, self.slice_count, 1])
        conv1 = tensorflow.nn.relu(self.conv_3d(variable_x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self.max_pool_3d(conv1)
        conv2 = tensorflow.nn.relu(self.conv_3d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self.max_pool_3d(conv2)
        fc = tensorflow.reshape(conv2, [-1, 54080])
        fc = tensorflow.nn.relu(tensorflow.matmul(fc, weights['W_fc'])+biases['b_fc'])
        fc = tensorflow.nn.dropout(fc, self.keep_rate)
        output = tensorflow.matmul(fc, weights['out'])+biases['out']
        return output

    def max_pool_3d(self, variable_x):
        """
        DOCSTRING
        """
        return tensorflow.nn.max_pool3d(
            variable_x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    def mean(self, list_a):
        """
        DOCSTRING
        """
        return sum(list_a) / len(list_a)

    def preprocess_data(self):
        """
        DOCSTRING
        """
        much_data = []
        for num, patient in enumerate(self.patients[:10]):
            if num % 100 == 0:
                print(num)
            try:
                img_data, label = self.process_data(
                    patient, self.labels, hm_slices=self.slice_count, image_size=self.image_size)
                much_data.append([img_data, label])
            except KeyError as exception:
                print('Error:Unlabeled Data')
        numpy.save('muchdata-{}-{}-{}.npy'.format(
            self.image_size, self.image_size, self.slice_count), much_data)

    def process_data(self, patient, labels, hm_slices=20, image_size=50, visualize=False):
        """
        DOCSTRING
        """
        for patient in self.patients[:10]:
            try:
                label = self.labels.get_value(patient, 'cancer')
                path = self.data_path + patient
                slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
                slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
                new_slices = []
                slices = [
                    cv2.resize(
                        numpy.array(each_slice.pixel_array),
                        (self.image_size, self.image_size)) for each_slice in slices]
                chunk_sizes = math.ceil(len(slices)/self.slice_count)
                for slice_chunk in self.chunks(slices, chunk_sizes):
                    slice_chunk = list(map(self.mean, zip(*slice_chunk)))
                    new_slices.append(slice_chunk)
                if len(new_slices) == self.slice_count-1:
                    new_slices.append(new_slices[-1])
                if len(new_slices) == self.slice_count-2:
                    new_slices.append(new_slices[-1])
                    new_slices.append(new_slices[-1])
                if len(new_slices) == self.slice_count+2:
                    new_val = list(map(self.mean, zip(*[new_slices[self.slice_count-1],
                                                        new_slices[self.slice_count],])))
                    del new_slices[self.slice_count]
                    new_slices[self.slice_count-1] = new_val
                if len(new_slices) == self.slice_count+1:
                    new_val = list(
                        map(self.mean, zip(*[new_slices[self.slice_count-1],
                                             new_slices[self.slice_count],])))
                    del new_slices[self.slice_count]
                    new_slices[self.slice_count-1] = new_val
                print(len(slices), len(new_slices))
            except Exception as exception:
                print(str(exception))
            if visualize:
                figure = pyplot.figure()
                for num, each_slice in enumerate(slices[:12]):
                    variable_y = figure.add_subplot(4, 5, num+1)
                    new_image = cv2.resize(
                        numpy.array(each_slice.pixel_array), (self.image_size, self.image_size))
                    variable_y.imshow(new_image)
                pyplot.show()
            if label == 1:
                label = numpy.array([0, 1])
            elif label == 0:
                label = numpy.array([1, 0])
            return numpy.array(new_slices), label

    def train_neural_network(self, variable_x):
        """
        DOCSTRING
        """
        prediction = self.convolutional_neural_network(variable_x)
        cost = tensorflow.reduce_mean(
            tensorflow.nn.softmax_cross_entropy_with_logits(prediction, self.variable_y))
        optimizer = tensorflow.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
        hm_epochs = 10
        with tensorflow.Session() as sess:
            sess.run(tensorflow.initialize_all_variables())
            successful_runs = 0
            total_runs = 0
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for data in self.train_data:
                    total_runs += 1
                    try:
                        X = data[0]
                        Y = data[1]
                        _, c = sess.run(
                            [optimizer, cost], feed_dict={self.variable_x: X, self.variable_y: Y})
                        epoch_loss += c
                        successful_runs += 1
                    except Exception as exception:
                        pass
                print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
                correct = tensorflow.equal(
                    tensorflow.argmax(prediction, 1), tensorflow.argmax(self.variable_y, 1))
                accuracy = tensorflow.reduce_mean(tensorflow.cast(correct, 'float'))
                print('Accuracy:',
                      accuracy.eval({self.variable_x:[i[0] for i in self.validation_data],
                                     self.variable_y:[i[1] for i in self.validation_data]}))
            print('Accuracy:',
                  accuracy.eval({self.variable_x:[i[0] for i in self.validation_data],
                                 self.variable_y:[i[1] for i in self.validation_data]}))
            print('Fitness Percent:', successful_runs/total_runs)

if __name__ == '__main__':
    ConvNet()
