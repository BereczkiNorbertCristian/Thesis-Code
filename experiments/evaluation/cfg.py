MODEL_PATH = '../../app/models/retinanet_resnet50_sign.h5'
VALIDATION_PATH = '../experiment 2/super_dataset/validation'
VALIDATION_PATH_ANNOTATIONS = '../experiment 2/super_dataset/validation/bboxes.csv'
BACKBONE = 'resnet50'
labels_to_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
                   12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
RESULTS_PATH = 'resnet50/results.out'
