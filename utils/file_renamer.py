import os

RAW_DATA_PATH = '../resources/face_data/raw'

# Changes raw names like '1.png' into '001_StudentName.png'


def __rename_file(path, old_name, dir_name):
    number = old_name[:len(old_name) - 4]
    new_file_name = number.zfill(3) + '_' + dir_name
    os.rename(path + '/' + old_name, path + '/' + new_file_name + '.png')


def __rename_files(raw_folder_path):
    for dir_name in os.listdir(raw_folder_path):
        full_dir_name = raw_folder_path + '/' + dir_name
        if (os.path.isdir(full_dir_name)):
            print('->', dir_name)

            for org_file_name in os.listdir(full_dir_name):
                if len(org_file_name) < 7:
                    __rename_file(full_dir_name, org_file_name, dir_name)


# Main func of file
def main_renamer(raw_folder_path=RAW_DATA_PATH):
    __rename_files(raw_folder_path)
