
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":

    test = pd.read_csv(config['DATA']['test_path'])

    model = joblib.load(config['MODELS']['sgd_model'])

    true_labels = test[config['MODELLING']['target']]
    pred_labels = model.predict(test)
    pred_probabilities = model.predict_proba(test)

    clf_report_df = pd.DataFrame(classification_report(
                true_labels,
                pred_labels,
                output_dict=True))
    clf_report_df.to_csv(config['SGD_EVAL']['classification_report_path'],
                         index=True,
                         index_label="metric")

    printed_report = classification_report(
                true_labels,
                pred_labels)
    print(f"\nSGD Classifier:\n\n{printed_report}")


    #create ROC curve
    display = metrics.RocCurveDisplay.from_estimator(
        model,
        test,
        test[config['MODELLING']['target']],
        name="SGD Classifier",
        pos_label="Android")

    display.figure_.savefig(config['SGD_EVAL']['roc_plot_path'])

    print("SGD classifier evaluation - DONE")