import os
import imageio.v2 as iio
import pandas as pd
import pickle
from PIL import Image


def resize_images(dataset_dir, data_path, new_size=False, verbose=False):
    """
    resize original image (256, 256) to new_size 
    save the new images to folder whose name added by _newsize
    save data (array of images and csv) to data_path
    """
    if new_size:
        new_dataset_dir = '_'.join(
            (dataset_dir, str(new_size[0]), str(new_size[1])))
    else:
        new_dataset_dir = '_'.join(
            (dataset_dir, str(256), str(256)))
    dataset = []
    dataset_info_df = pd.DataFrame()
    if not os.path.exists(new_dataset_dir):
        os.mkdir(new_dataset_dir)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    for ethnicity in range(13):
        for gender in range(2):
            idx = 0
            png_dir = ''.join(
                (dataset_dir, '/', f'{ethnicity:02}', str(gender), f'{idx:04}', '.png'))
            csv_dir = ''.join(
                (dataset_dir, '/', f'{ethnicity:02}', str(gender), f'{idx:04}', '.csv'))
            while os.path.exists(png_dir):
                try:
                    if new_size:
                        image = (Image.open(png_dir)).resize(new_size)
                        png_dir = ''.join(
                            (new_dataset_dir, '/', f'{ethnicity:02}', str(gender), f'{idx:04}', '.png'))
                        image.save(png_dir)
                    dataset.append(iio.imread(png_dir))
                    new_info_df = pd.read_csv(csv_dir)
                    dataset_info_df = pd.concat(
                        [dataset_info_df, new_info_df], ignore_index=True)
                except:
                    if verbose:
                        # broken image? ignore
                        print(
                            f'Image {ethnicity:02}{gender}{idx:04} discarded.')
                idx += 1
                csv_dir = ''.join(
                    (dataset_dir, '/', f'{ethnicity:02}', str(gender), f'{idx:04}', '.csv'))
                png_dir = ''.join(
                    (dataset_dir, '/', f'{ethnicity:02}', str(gender), f'{idx:04}', '.png'))
        if verbose:
            print(f'Ethnicity {ethnicity} finished.')

    with open(''.join((data_path, '/dataset.pkl')), 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(''.join((data_path, '/dataset_info.pkl')), 'wb') as handle:
        pickle.dump(dataset_info_df, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    DATASET_DIR = './BatchComposites'
    DATA_PATH = './Data'
    NEW_SIZE = (64, 64)
    VERBOSE = True
    resize_images(DATASET_DIR, DATA_PATH, NEW_SIZE, VERBOSE)
