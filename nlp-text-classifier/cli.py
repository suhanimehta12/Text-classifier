"""
cli.py
------
Command-line interface for the NLP text classifier.

Commands:
  train     — Train a model from a CSV file
  evaluate  — Evaluate a saved model on a dataset
  predict   — Classify one text string or a batch from a file
"""

import click
from src.train import train as run_train
from src.evaluate import evaluate as run_evaluate
from src.predict import load_model, predict_text, predict_from_file, format_result


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """🧠 NLP Text Classifier — NLTK + scikit-learn"""
    pass


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data",       required=True,              help="Path to training CSV file.")
@click.option("--model-out",  default="models/classifier.pkl", show_default=True, help="Where to save the trained model.")
@click.option("--model-type", default="logistic",         show_default=True,
              type=click.Choice(["logistic", "naive_bayes", "svm", "random_forest"]),
              help="Classifier to use.")
@click.option("--text-col",   default="text",             show_default=True, help="Name of the text column in the CSV.")
@click.option("--label-col",  default="label",            show_default=True, help="Name of the label column in the CSV.")
@click.option("--test-size",  default=0.2,                show_default=True, help="Fraction held out for testing.")
def train(data, model_out, model_type, text_col, label_col, test_size):
    """Train a text classification model."""
    result = run_train(
        data_path=data,
        model_out=model_out,
        model_type=model_type,
        text_col=text_col,
        label_col=label_col,
        test_size=test_size,
    )
    click.echo(f"\n✅ Training complete!")
    click.echo(f"   Model type  : {result['model_type']}")
    click.echo(f"   Train acc   : {result['train_acc']:.4f}")
    click.echo(f"   Classes     : {result['n_classes']}")
    click.echo(f"   Train/Test  : {result['n_train']} / {result['n_test']}")


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data",      required=True,              help="Path to CSV file (same format as training).")
@click.option("--model",     default="models/classifier.pkl", show_default=True, help="Path to saved model.")
@click.option("--text-col",  default="text",             show_default=True)
@click.option("--label-col", default="label",            show_default=True)
@click.option("--no-plot",   is_flag=True,               help="Skip saving the confusion matrix plot.")
@click.option("--cm-out",    default="models/confusion_matrix.png", show_default=True,
              help="Where to save the confusion matrix image.")
def evaluate(data, model, text_col, label_col, no_plot, cm_out):
    """Evaluate a trained model and print a classification report."""
    run_evaluate(
        model_path=model,
        data_path=data,
        text_col=text_col,
        label_col=label_col,
        plot_cm=not no_plot,
        cm_out=cm_out,
    )


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--text",   default=None, help="Single text string to classify.")
@click.option("--file",   default=None, help="Path to a .txt file (one text per line) for batch prediction.")
@click.option("--model",  default="models/classifier.pkl", show_default=True, help="Path to saved model.")
def predict(text, file, model):
    """Classify a single text or a batch from a file."""
    if not text and not file:
        raise click.UsageError("Provide either --text or --file.")

    pipeline = load_model(model)

    if text:
        result = predict_text(text, pipeline)
        click.echo(f"\n{'─'*50}")
        click.echo(format_result(result))
        click.echo(f"{'─'*50}\n")

    elif file:
        results = predict_from_file(file, pipeline)
        click.echo(f"\n{'─'*50}")
        for i, res in enumerate(results, 1):
            click.echo(f"[{i}]")
            click.echo(format_result(res))
        click.echo(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
