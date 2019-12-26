import os

RAW_DATA_PATH = '../resources/face_data/raw'

# Changes raw names like '1.png' into '001_StudentName.png'


def rename_file(path, old_name, dir_name):
    number = old_name[:len(old_name) - 4]
    new_file_name = number.zfill(3) + '_' + dir_name
    os.rename(path + '/' + org_file_name, path + '/' + new_file_name + '.png')


for dir_name in os.listdir(RAW_DATA_PATH):
    full_dir_name = RAW_DATA_PATH + '/' + dir_name
    if (os.path.isdir(full_dir_name)):
        print('->', dir_name)

        for org_file_name in os.listdir(full_dir_name):
            if len(org_file_name) < 7:
                rename_file(full_dir_name, org_file_name, dir_name)
