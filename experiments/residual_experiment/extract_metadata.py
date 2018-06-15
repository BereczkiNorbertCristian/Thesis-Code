
import os
import random

H = {}

DATASET_FOLDER = 'raw_dataset'
VALIDATION_SAMPLE = 400

# for sign_maker in os.listdir(DATASET_FOLDER):
#     SIGN_MAKER_FOLDER = os.path.join(DATASET_FOLDER, sign_maker)
#     for sign in os.listdir(SIGN_MAKER_FOLDER):
#         SIGN_SUBFOLDER = os.path.join(SIGN_MAKER_FOLDER, sign)
#         for image_fname in os.listdir(SIGN_SUBFOLDER):
#             if sign not in H:
#                 H[sign] = []
#             H[sign].append((sign_maker, sign, image_fname))

# signs = list(H.keys())
# signs.sort()
# train_lst = []
# val_lst = []

# with open('class_to_id.csv', 'w') as f:
#     for ID, sign in enumerate(signs):
#         f.write(str(ID) + "," + str(sign) + "\n")

# for sign in signs:
#     val_fraction = random.sample(H[sign], VALIDATION_SAMPLE)
#     train_fraction = [el for el in H[sign] if el not in val_fraction]
#     train_lst.extend(train_fraction)
#     val_lst.extend(val_fraction)

sign_makers = ['A','B','C','D']

train_lst = []
val_lst = []

for sign_maker in sign_makers:
    SIGN_MAKER_FOLDER = os.path.join(DATASET_FOLDER, sign_maker)
    for sign in os.listdir(SIGN_MAKER_FOLDER):
        SIGN_SUBFOLDER = os.path.join(SIGN_MAKER_FOLDER, sign)
        for image_fname in os.listdir(SIGN_SUBFOLDER):
            train_lst.append((sign_maker, sign, image_fname))

signs = []
lst = []
VAL_SIGN_MAKER_FOLDER = os.path.join(DATASET_FOLDER,'E')
for sign in os.listdir(VAL_SIGN_MAKER_FOLDER):
    SIGN_SUBFOLDER = os.path.join(VAL_SIGN_MAKER_FOLDER, sign)
    signs.append(sign)
    for image_fname in os.listdir(SIGN_SUBFOLDER):
        val_lst.append((sign_maker, sign, image_fname))

random.shuffle(train_lst)
random.shuffle(val_lst)

print(len(val_lst))

with open('train.csv', 'w') as f:
    for ID, el in enumerate(train_lst):
        f.write(str(ID) + "," + str(el[0]) +
                "," + str(el[1]) + "," + str(el[2]) + "\n")

with open('validation.csv', 'w') as f:
    for ID, el in enumerate(val_lst):
        f.write(str(ID) + "," + str(el[0]) +
                "," + str(el[1]) + "," + str(el[2]) + "\n")

# print("Stats: ")
# sm = 0
# sm_perc = 0
# for sign in signs:
#     ln = len(H[sign])
#     print(sign + " number of images: " + str(ln))
#     print(sign + " perc of test: " + str(VALIDATION_SAMPLE / ln))
#     sm += ln
#     sm_perc += VALIDATION_SAMPLE / ln
# print("Avg number of images per sign: " + str(sm / 24))
# print("Avg perc of test fraction per sign: " + str(sm_perc / 24))
