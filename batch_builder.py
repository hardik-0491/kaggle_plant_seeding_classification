import prepare_dataset
from PIL import Image

class BatchBuilder:
    data: list
    batch_size: int
    start_index: int

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.initialize_batches()

    def initialize_batches(self):
        self.start_index = 0

    def get_next_batch(self):
        batch_x = []
        batch_y = []

        if self.start_index < len(self.data):

            if self.start_index + self.batch_size < len(self.data):
                end_index = self.start_index + self.batch_size
            else:
                end_index = len(self.data) - 1

            for index in range(self.start_index, end_index, 1):

                batch_data = prepare_dataset.prepare_training_dataset(self.data[index])
                batch_x.append(batch_data['input'])
                batch_y.append(batch_data['output'])

            self.start_index = end_index + 1

        return batch_x, batch_y

    def data_exist(self):

        if self.start_index < len(self.data):
            return True
        else:
            return False