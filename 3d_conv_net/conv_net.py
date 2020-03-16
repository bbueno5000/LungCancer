"""
DOCSTRING
"""
# standard
import os
# non-standard
import cv2
import pydicom
import math
import matplotlib.pyplot as pyplot
import numpy
import pandas

class ConvNet:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        self.data_path = 'C:\\Downloads\\input\\sample_images'
        self.labels = pandas.read_csv(
            'C:\\Users\\bbuen\\Downloads\\DSB3\\stage1_labels.csv', index_col=0)
        self.slice_count = 20
        self.image_size = 150 # pixels
        self.patients = os.listdir(self.data_path)

    def __call__(self):
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
                much_data.append([img_data,label])
            except KeyError as exception:
                print('Error:Unlabeled Data')
        numpy.save('muchdata-{}-{}-{}.npy'.format(
            self.image_size, self.image_size, self.slice_count), much_data)

    def chunks(self, l, n):
        """
        Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def mean(self, l):
        """
        DOCSTRING
        """
        return sum(l) / len(l)

    def process_data(self, patient, labels, hm_slices=20, image_size=50, visualize=False):
        """
        DOCSTRING
        """
        for patient in self.patients[:10]:
            try:
                label = self.labels.get_value(patient, 'cancer')
                path = self.data_path + patient
                slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
                slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
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
                    new_val = list(
                        map(self.mean, zip(*[new_slices[self.slice_count-1], new_slices[self.slice_count],])))
                    del new_slices[self.slice_count]
                    new_slices[self.slice_count-1] = new_val
                if len(new_slices) == self.slice_count+1:
                    new_val = list(
                        map(self.mean, zip(*[new_slices[self.slice_count-1], new_slices[self.slice_count],])))
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
                label=numpy.array([0, 1])
            elif label == 0: 
                label=numpy.array([1, 0])
            return numpy.array(new_slices), label

if __name__ == '__main__':
    ConvNet()
