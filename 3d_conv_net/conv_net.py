"""
DOCSTRING
"""
# standard
import os
# non-standard
import cv2
import pydicom
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
        self.data_path = 'C:/Downloads/input/sample_images/'
        self.patients = os.listdir(self.data_path)
        self.dataframe_a = pandas.read_csv('C:/Downloads/input/stage1_labels.csv', index_col=0)

    def load_data(self):
        """
        DOCSTRING
        """
        for patient in self.patients[:1]:
            label = self.dataframe_a.get_value(patient, 'cancer')
            path = self.data_path + patient
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            print(len(slices), label)
            print(slices[0])
        for patient in self.patients[:3]:
            label = self.dataframe_a.get_value(patient, 'cancer')
            path = self.data_path + patient
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            print(slices[0].pixel_array.shape, len(slices))

    def scale(self):
        """
        DOCSTRING
        """
        image_size = 150 # pixels
        for patient in self.patients[:1]:
            label = self.dataframe_a.get_value(patient, 'cancer')
            path = self.data_path + patient
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            figure = pyplot.figure()
            for num, each_slice in enumerate(slices[:12]):
                variable_y = figure.add_subplot(3, 4, num+1)
                new_image = cv2.resize(numpy.array(each_slice.pixel_array), (image_size, image_size))
                variable_y.imshow(new_image)
            pyplot.show()

    def visualize(self):
        """
        DOCSTRING
        """
        for patient in self.patients[:1]:
            label = self.dataframe_a.get_value(patient, 'cancer')
            path = self.data_path + patient
            slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            pyplot.imshow(slices[0].pixel_array)
            pyplot.show()

if __name__ == '__main__':
    ConvNet().visualize()
