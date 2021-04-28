import numpy as np
import matplotlib.pyplot as plt

from tools.general.json_utils import read_json

plt.clf()

"""
OGQ-3M 2314986 12499
OPIV6-1,2 1407495 10717
OPIV6-3 1301000 10772
OPIV6-4 489494 10306
OPIV6-Validation 37539 9905
OPIV6-Test 108144 10625
"""
for tag, path, color in [
        ['OGQ-3M', './data/OGQ_3M.json', 'b'],
        ['OPIV6-1,2', './data/OPIV6-2M-1,2.json', 'g'],
        # ['OPIV6-3', './data/OPIV6-2M-3.json', 'r'],
        # ['OPIV6-4', './data/OPIV6-2M-4.json', 'r'],
        
        # ['OPIV6-Validation', './data/OPIV6-Validation.json', 'c'],
        # ['OPIV6-Test', './data/OPIV6-Test.json', 'm'],
    ]:
    data_dic = read_json(path)

    number_of_images = data_dic['number_of_images']
    classes = data_dic['classes']
    
    class_names = data_dic['class_names']
    count_per_class = [data_dic['count_dic'][class_name] for class_name in class_names]

    count_per_class = np.log(count_per_class)

    print(tag, number_of_images, classes)

    plt.plot(np.arange(len(class_names)), sorted(count_per_class)[::-1], color, label=tag)
    # plt.plot(np.arange(len(class_names)), count_per_class, color, label=tag)

# plt.title('{} - number of images : {:,}'.format(tag, number_of_images))
plt.legend()
plt.ylabel('Log(x)')
plt.xlabel('Class Index')
plt.show()

