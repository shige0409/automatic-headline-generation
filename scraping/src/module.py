import requests
import config


class RequestDriver():
    def __init__(self) -> None:
        self.page_source = None
    def get(self, url):
        response = requests.get(
            url,
            headers=config.headers,
            proxies=config.proxy
            )
        self.page_source = response.content

class Slacker():
    def post_message(self, text):
        res = requests.post(
            url=config.POST_MESSAGE_URL,
            data={
                "token": config.TOKEN,
                "channel": config.CHANNEL,
                "text": text
            }
        )

    def post_file(self, file_path):
        with open(file_path, "rb") as f:
            res = requests.post(
                url=config.FILE_UPLOAD_URL,
                data={
                    "token": config.TOKEN,
                    "channels": config.CHANNEL,
                },
                files={"file": f}
            )
            print(res.json())