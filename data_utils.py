import os
import shutil


def prep_directory():
    def make_dir(dir_name):
        if os.path.exists(dir_name):
            os.rename(dir_name, dir_name + '_backup')
            shutil.rmtree(dir_name + '_backup')

        os.mkdir(dir_name)

    make_dir('data')
    make_dir('output')


def download_data():
    print('Starting video downloads')


def main():
    prep_directory()

    input('Press Enter after Adding train.csv and test.csv in the data folder')

    if not os.path.exists('data/avspeech_train.csv') or not os.path.exists('data/avspeech_test.csv'):
        print('No training and Testing data found')
    else:
        download_data()


if __name__ == "__main__":
    main()
