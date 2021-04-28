import sys
import glob
import argparse
import numpy as np

from core.datasets import *
from core.opiv6_modules import get_dictionary

from tools.general.string_utils import is_english
from tools.general.json_utils import write_json

parser = argparse.ArgumentParser()
# parser.add_argument('--root_dir', default='../', type=str) # '//192.168.100.62/gynetworks/'
# parser.add_argument('--dataset', default='OGQ_3M', type=str)  # 'train', 'validation', 'test', '*'
# parser.add_argument('--dataset_name', default='OGQ-3M_SH', type=str)
# parser.add_argument('--domain', default='train_all', type=str)  # 'train', 'validation', 'test', '*'
# parser.add_argument('--json_name', default='OPIV6.json', type=str)
# parser.add_argument('--source', default='labels_from_machine', type=str)

parser.add_argument('--root_dir', default='//192.168.100.62/gynetworks/data/', type=str) # '//192.168.100.62/gynetworks/'
parser.add_argument('--dataset', default='', type=str)  # 'train', 'validation', 'test', '*'
parser.add_argument('--dataset_name', default='PIXTA_SH', type=str)
parser.add_argument('--domain', default='validation', type=str)  # 'train', 'validation', 'test', '*'
parser.add_argument('--json_name', default='PIXTA-18M.json', type=str)
parser.add_argument('--source', default='labels_from_machine', type=str)

args = parser.parse_args()

def refine_tags(tags):
    refined_tags = []
    for tag in tags:
        if len(tag) == 0:
            continue
        elif tag[0] == ' ':
            continue

        try:
            float(tag[0])
            continue
        except ValueError:
            pass
        
        refined_tags.append(tag)
    return refined_tags

class Dataset_For_Tags(SH_Dataset):
    def __init__(self, data_patterns, data_dic, transform=None):
        file_paths = []
        for data_pattern in data_patterns:
            file_paths += glob.glob(data_pattern)

        super().__init__(file_paths, transform, debug=True)

        self.description = get_dictionary('./data/oidv6-class-descriptions.csv')
        
        if data_dic is None:
            self.class_names = []
            self.class_dic = {}
            self.classes = 0
        else:
            self.class_names = np.asarray(data_dic['class_names'])
            self.class_dic = {name : index for index, name in enumerate(self.class_names)}
            self.classes = data_dic['classes']
        
        self.source = args.source
    
    def decode(self, example):
        if 'labels' in self.source:
            tags = []
            for encoded_name in example[self.source]:
                try:
                    name = self.description[encoded_name].lower()

                    if len(class_names) > 0:
                        if name in self.class_names:
                            tags.append(name)
                    else:
                        tags.append(name)

                except KeyError:
                    pass
        else:
            tags = example[self.source]
        
        if 'PIXTA' in args.dataset_name:
            tags = refine_tags(tags)

        return tags

args.domain = args.domain.replace('all', '*')

data_patterns = [
    args.root_dir + f'{args.dataset_name}/{args.domain}/*.sang', 
]

print(data_patterns)

if args.dataset == '':
    data_dic = None
else:
    data_dic = read_json(f'./data/{args.dataset}.json', encoding='utf-8')

dataset = Dataset_For_Tags(data_patterns, data_dic)

print(len(dataset))

count_dic = {}

for i, tags in enumerate(dataset):
    sys.stdout.write(f'\r[{i+1}/{len(dataset)}], {len(count_dic.keys())}')
    sys.stdout.flush()

    # add tags
    for tag in tags:
        # print(tag)

        if args.dataset != '':
            if not is_english(tag):
                print('skip : {}'.format(tag))
                continue
        
        try:
            count_dic[tag] += 1
        except KeyError:
            count_dic[tag] = 1

if args.dataset == '':
    class_names = list(count_dic.keys())
    before_length = len(class_names)
    
    for class_name in class_names:
        if count_dic[class_name] < 10:
            # print('[DELETE] {}'.format(class_name))
            del count_dic[class_name]

    after_length = len(list(count_dic.keys()))
    print('[DELETE]', before_length, '->', after_length)

class_names = sorted(list(count_dic.keys()))

write_json('./data/' + args.json_name, {
    'number_of_images' : len(dataset),
    'count_dic' : count_dic,
    'class_names' : class_names,
    'classes' : len(class_names)
})