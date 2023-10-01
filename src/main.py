import click

from src.model.train_model import train_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    train_model(input_filename, model_dump_filename)


cli.add_command(train)


if __name__ == "__main__":
    cli()
