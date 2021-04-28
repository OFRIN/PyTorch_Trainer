# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os

from tools.data.utils import *
from tools.general.io_utils import *

class SH_Writer:
    def __init__(self, data_dir, data_pattern, the_number_of_example_per_file):
        self.data_index = 1
        self.data_format = create_directory(data_dir) + data_pattern
        
        self.the_number_of_example_per_file = the_number_of_example_per_file
        
        self.start()
    
    def start(self):
        self.data_path, self.index_path = self.get_path()
        
        self.data_f = open(self.data_path, 'wb')
        self.index_f = open(self.index_path, 'w')

        self.accumulated_size = 0
    
    def end(self):
        self.data_f.close()
        self.index_f.close()

        if self.accumulated_size == 0:
            os.remove(self.data_path)
            os.remove(self.index_path)
        
    def __call__(self, example):
        start_point = self.data_f.tell()
        bytes_of_example = serialize(example)
        length_of_example = len(bytes_of_example)
        
        self.data_f.write(bytes_of_example)
        self.index_f.write(f'{start_point},{length_of_example}\n')

        self.accumulated_size += 1
        if self.accumulated_size == self.the_number_of_example_per_file:
            self.end()
            self.start()

    def get_path(self):
        data_path = self.data_format.format(self.data_index)
        index_path = data_path.replace('.sang', '.index')

        self.data_index += 1
        return data_path, index_path
        