import sys
from pathlib import Path
import os.path

my_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + "/../digital-pathology-master/src/SlideAnalysis/src/py"

sys.path.insert(0, my_path)
import predictor

if __name__ == '__main__':
    p = predictor.Predict()
    classes1 = p.predict_img_class_folder("../cellData/eosinophil")
    classes2 = p.predict_img_class_folder("../cellData/non-eosinophil")
    #print(classes)
    correct_rate_ecell = len([k for k,v in classes1.items() if v == 'ecell'])/len(classes1.keys())
    print('Rate of correctly classify eosinophil cell: '+str(correct_rate_ecell))
    correct_rate_nonecell = len([k for k,v in classes2.items() if v == 'non-ecell'])/len(classes2.keys())
    print('Rate of correctly classify non-eosinophil cell: '+str(correct_rate_nonecell))
