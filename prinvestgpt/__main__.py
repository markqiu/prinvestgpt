import logging
import platform

import typer
import uvicorn

from prinvestgpt.__about__ import __version__
from prinvestgpt.client import corpus, server
from prinvestgpt import settings


app = typer.Typer()
app.add_typer(corpus.app, name="corpus")
app.add_typer(server.app, name="server")

if platform.system().lower() == "windows":
    OSTYPE = "windows"
elif platform.system().lower() == "linux":
    OSTYPE = "linux"
elif platform.system().lower() == "darwin":
    OSTYPE = "macos"


@app.command()
def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        "reworkd_platform.web.application:get_app",
        workers=settings["workers_count"],
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        factory=True,
    )


@app.command()
def version():
    """
    显示版本号
    """
    logging.info(__version__)


if __name__ == "__main__":
    app()
