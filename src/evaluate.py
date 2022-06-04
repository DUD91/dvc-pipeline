import typer
import tensorflow as tf

from mdutils.mdutils import MdUtils
from mdutils import Html

def generate_evaluation_report():
    X_test =tf.io.read_file("data/X_test", name=None)
    Y_test = tf.io.read_file("data/Y_test", name=None)

    model = tf.keras.models.load_model('models/custom_cnn')

    print("Evaluating model on test set")
    loss_test, metric_test = model.evaluate(X_test, Y_test, verbose=0)

    mdFile = MdUtils(file_name='Model_Evaluation_Report', title='Model Evaluation Report')
    mdFile.new_header(level=1, title='Test Evaluation')

    mdFile.new_paragraph(
        f"The model trained on Cifar 10 has the following evaluation metrics:"
        f"Test loss: {round(loss_test, 4)}"
        f"Test Accuracy: {round(metric_test,4)}")

    print("Generating Markdown File")
    mdFile.create_md_file()

if __name__ == '__main__':
    typer.run(generate_evaluation_report)


