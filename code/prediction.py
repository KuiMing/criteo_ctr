from utils import predict
import click


@click.command()
@click.option("-r", "result_path", help="result path")
@click.option("-e", "encoder_path", help="encoding path")
@click.option("-m", "model_path", help="model path")
@click.option("-d", "data_path", help="data path")
def main(data_path, encoder_path, model_path, result_path):
    predict(data_path, encoder_path, model_path, result_path)


if __name__ == "__main__":
    main()
