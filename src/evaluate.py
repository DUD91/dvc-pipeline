import os

import typer
import tensorflow as tf
import numpy as np
from utils import load_dvc_params


from mdutils.mdutils import MdUtils

def generate_evaluation_report():
    params = load_dvc_params()
    X_test = np.load(params["X_test"])
    Y_test = np.load(params["Y_test"])

    model = tf.keras.models.load_model(params['model_dir'])

    print("Evaluating model on test set")
    loss_test, metric_test = model.evaluate(X_test, Y_test, verbose=0)

    mdFile = MdUtils(file_name='Model_Evaluation_Report', title='Model Evaluation Report')
    mdFile.new_header(level=1, title='Test Evaluation')

    mdFile.new_paragraph(
        f"The model trained on Cifar 10 has the following evaluation metrics: \n\n"
        f"Test loss: {round(loss_test, 4)} \n\n"
        f"Test Accuracy: {round(metric_test,4)}")

    print("Generating Markdown File")
    mdFile.create_md_file()
    os.rename('Model_Evaluation_Report.md', f"{params['eval_dir']}Model_Evaluation_Report.md")

if __name__ == '__main__':
    typer.run(generate_evaluation_report)


