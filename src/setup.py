import os

# CREATING FOLDERS

folders_to_create = ['bin/']

for folder in folders_to_create:
    print('Folder ' + folder + ' ', end='')
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('is created!')
    else:
        print('already exists!')