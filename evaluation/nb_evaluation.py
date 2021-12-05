
from sklearn.metrics import classification_report
from sklearn import metrics
import pandas as pd
import joblib

import configparser
config = configparser.ConfigParser()
config.read("config.ini")

if __name__ == "__main__":

    test = pd.read_csv(config['DATA']['test_path'])

    model = joblib.load(config['MODELS']['nb_model'])

    true_labels = test[config['MODELLING']['target']]
    pred_labels = model.predict(test)
    pred_probabilities = model.predict_proba(test)

    clf_report_df = round(pd.DataFrame(classification_report(
                true_labels,
                pred_labels,
                output_dict=True)), 3)
    clf_report_df.to_csv(config['NB_EVAL']['classification_report_path'],
                         index=True,
                         index_label="metric")

    printed_report = classification_report(
                true_labels,
                pred_labels)
    print(f"\nNB Classifier:\n\n{printed_report}")

    #create ROC curve
    roc = metrics.RocCurveDisplay.from_estimator(
        model,
        test,
        test[config['MODELLING']['target']],
        name="NB Classifier",
        pos_label="Android")
    roc.figure_.savefig(config['NB_EVAL']['roc_plot_path'])

    #create confusion matrix
    cm = metrics.ConfusionMatrixDisplay.from_estimator(
        model,
        test,
        test[config['MODELLING']['target']],
        display_labels=model.classes_)
    cm.figure_.savefig(config['NB_EVAL']['cm_plot_path'])

    print("NB classifier evaluation - DONE")