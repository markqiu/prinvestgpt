import logging
import platform

import typer
from dotenv import load_dotenv

from prinvestgpt.__about__ import __version__
from prinvestgpt.client import corpus, server

app = typer.Typer()
app.add_typer(corpus.app, name="corpus")
app.add_typer(server.app, name="server")


load_dotenv()  # take environment variables from .env.


if platform.system().lower() == "windows":
    OSTYPE = "windows"
elif platform.system().lower() == "linux":
    OSTYPE = "linux"
elif platform.system().lower() == "darwin":
    OSTYPE = "macos"


@app.command()
def version():
    """
    显示版本号
    """
    logging.info(__version__)


if __name__ == "__main__":
    app()
