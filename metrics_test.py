import pandas as pd
# from precisionweighted import weighted_precision
# from recallweighted import weighted_recall
from weightedmetrics import *
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices', 'Hernia', 'Mass',
        'Fibrosis', 'Infiltration', 'Nodule', 'Emphysema', 'Pleural_Thickening']

filename = './gtfile.csv'
filename2 = './predfile.csv'
df = pd.read_csv(filename)
df2 = pd.read_csv(filename2)
gt = df.to_numpy()
pred = df2.to_numpy()
for i, (name) in enumerate(class_names):
    print(name)
    weighted_average_precision = weighted_precision(gt[:, i], pred[:, i])
    weighted_average_recall = weighted_recall(gt[:, i], pred[:, i])
    f1score = weighted_f1score(gt[:, i], pred[:, i])
    accuracy_1 = accuracy_sc(gt[:, i], pred[:, i])
    weighted_average_specificity = weighted_specificity(gt[:, i], pred[:, i])
    print('weighted_average_precision : ', weighted_average_precision)
    print('weighted_average_recall : ', weighted_average_recall)
    print('f1score : ', f1score)
    print('accuracy', accuracy_1)
    print('weighted_average_specificity', weighted_average_specificity)
