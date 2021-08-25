import requests
import config
import utils

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class RequestDriver():
    def __init__(self) -> None:
        self.page_source = None
        self.slacker = Slacker()

    def get(self, url):
        response = requests.get(
            url,
            headers=config.headers,
            proxies=config.proxy
        )
        utils.rand_sleep()
        self.page_source = response.content


class WebDriver():
    def __init__(self) -> None:
        op = Options()
        op.add_argument("enable-automation")
        op.add_argument("--no-sandbox")
        op.add_argument("--disable-infobars")
        op.add_argument('--disable-extensions')
        op.add_argument("--disable-dev-shm-usage")
        op.add_argument("--disable-browser-side-navigation")
        op.add_argument('--ignore-certificate-errors')
        op.add_argument('--ignore-ssl-errors')
        op.add_argument("--disable-gpu")
        op.add_argument("--disable-extensions")
        op.add_argument("--proxy-server='direct://'")
        op.add_argument("--proxy-bypass-list=*")
        op.add_argument("--start-maximized")
        op.add_argument("--headless")
        prefs = {"profile.default_content_setting_values.notifications": 2}
        op.add_experimental_option("prefs", prefs)
        self.driver = webdriver.Chrome(options=op)
        self.slacker = Slacker()

    def get(self, url):
        cannnot_get_page = True
        # ページをリクエストできるまで
        while(cannnot_get_page):
            try:
                self.driver.get(url)
                self.page_source = self.driver.page_source
                cannnot_get_page = False
                utils.rand_sleep()
            except Exception as e:
                self.slacker.post_message(e)
                self.slacker.post_message("エラーによりスクレイピングを10分休止します")
                # エラーが起きたら10分休憩する
                utils.rand_sleep(600, is_random=False)


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
