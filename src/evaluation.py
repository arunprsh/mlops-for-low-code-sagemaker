from sklearn.metrics import f1_score
import pandas as pd
import logging
import pathlib
import json
import os


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == '__main__':
    logger.info('Starting evaluation step')
    y_pred_path = '/opt/ml/processing/input/predictions/holdout.csv.out'
    y_pred = pd.read_csv(y_pred_path, header=None)
    y_true_path = '/opt/ml/processing/input/true_labels/true_labels.csv'
    y_true = pd.read_csv(y_true_path, header=None)
    
    report_dict = {
        'classification_metrics': {
            'weighted_f1': {
                'value': f1_score(y_true, y_pred, average='weighted'),
                'standard_deviation': 'NaN',
            },
        },
    }
    
    output_dir = '/opt/ml/processing/evaluation'
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, 'evaluation_metrics.json')
    
    with open(evaluation_path, 'w') as f:
        f.write(json.dumps(report_dict))
