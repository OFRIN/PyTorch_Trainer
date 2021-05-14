import ray
import copy
import argparse
import multiprocessing as mp

from google_images_download import google_images_download
from tools.general.json_utils import read_json

@ray.remote
def update(separated_class_names, limit):
    response = google_images_download.googleimagesdownload()

    for class_name in separated_class_names:
        
        arguments = {
            "keywords":class_name,
            "limit":limit,
            "chromedriver":"./data/chromedriver.exe"
        }
        try:
            paths = response.download(arguments)
        except:
            f = open('./data/error.txt', 'a', encoding='utf-8')
            f.write(class_name + '\n')
            f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--tags', default='car', type=str)
parser.add_argument('--num_imgs_per_tag', default=50, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    # cores = mp.cpu_count()
    cores = 6
    ray.init(num_cpus=cores)

    number_of_image_per_class = args.num_imgs_per_tag
    class_names = args.tags.split(',')
    
    ids = []
    size = len(class_names) // cores

    print('# recognized cores : {}'.format(cores))
    print('# size per core : {}'.format(size))

    for i in range(cores - 1):
        # ids.append(class_names[:size])
        ids.append(update.remote(copy.deepcopy(class_names[:size]), number_of_image_per_class))
        class_names = class_names[size:]

    # ids.append(class_names)
    ids.append(update.remote(class_names, number_of_image_per_class))

    # print()
    # c = 0
    # for id in ids:
    #     print(len(id), id[:5])
    #     c += len(id)
    # print(c)

    ray.get(ids)
