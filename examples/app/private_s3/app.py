#!pip install fsspec s3fs --upgrade
#!pip freeze
import lightning as L
import os, fsspec

class BoringWork(L.LightningWork):
    def run(self):
        with fsspec.open(
            urlpath="s3://your-private-bucket/your_file.txt",
            key=os.getenv("MY_AWS_ACCESS_KEY_ID"),
            secret=os.getenv("MY_AWS_SECRET_ACCESS_KEY"),
            token=os.getenv("MY_AWS_SESSION_TOKEN"),
        ).open() as f:
            print(f.readlines())

class BoringApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.work = BoringWork()

    def run(self):
        self.work.run()

app = L.LightningApp(BoringApp())
