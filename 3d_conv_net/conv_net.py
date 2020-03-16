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
        self.data_path = 'C:/Downloads/input/sample_images/'
        self.dataframe_a = pandas.read_csv(
            'C:\\Users\\bbuen\\Downloads\\DSB3\\stage1_labels.csv', index_col=0)
        self.hm_slices = 20
        self.image_size = 150 # pixels
        self.patients = os.listdir(self.data_path)

    def __call__(self):
        """
        DOCSTRING
        """
        for patient in self.patients[:10]:
            try:
                label = self.dataframe_a.get_value(patient, 'cancer')
                path = self.data_path + patient
                slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
                slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
                new_slices = []
                slices = [
                    cv2.resize(
                        numpy.array(each_slice.pixel_array),
                        (self.image_size, self.image_size)) for each_slice in slices]
                chunk_sizes = math.ceil(len(slices)/self.hm_slices)
                for slice_chunk in self.chunks(slices, chunk_sizes):
                    slice_chunk = list(map(self.mean, zip(*slice_chunk)))
                    new_slices.append(slice_chunk)
                if len(new_slices) == self.hm_slices-1:
                    new_slices.append(new_slices[-1])
                if len(new_slices) == self.hm_slices-2:
                    new_slices.append(new_slices[-1])
                    new_slices.append(new_slices[-1])
                if len(new_slices) == self.hm_slices+2:
                    new_val = list(
                        map(self.mean, zip(*[new_slices[self.hm_slices-1], new_slices[self.hm_slices],])))
                    del new_slices[self.hm_slices]
                    new_slices[self.hm_slices-1] = new_val
                if len(new_slices) == self.hm_slices+1:
                    new_val = list(
                        map(self.mean, zip(*[new_slices[self.hm_slices-1], new_slices[self.hm_slices],])))
                    del new_slices[self.hm_slices]
                    new_slices[self.hm_slices-1] = new_val
                print(len(slices), len(new_slices))
            except Exception as exception:
                print(str(exception))
            figure = pyplot.figure()
            for num, each_slice in enumerate(slices[:12]):
                variable_y = figure.add_subplot(4, 5, num+1)
                new_image = cv2.resize(
                    numpy.array(each_slice.pixel_array), (self.image_size, self.image_size))
                variable_y.imshow(new_image)
            pyplot.show()

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

if __name__ == '__main__':
    ConvNet()
