import os

# これがないとスクレピングできないサイトのため
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
    "refer": "https://news.livedoor.com/"
}
proxy = {
    "http": "socks5://127.0.0.1:9050",
    "https": "socks5://127.0.0.1:9050"}

# スクレイピング対象URL
category_urls = [
    'https://news.livedoor.com/topics/category/main/',
    'https://news.livedoor.com/topics/category/dom/',
    'https://news.livedoor.com/topics/category/world/',
    'https://news.livedoor.com/topics/category/eco/',
    'https://news.livedoor.com/topics/category/ent/',
    'https://news.livedoor.com/topics/category/sports/',
    'https://news.livedoor.com/topics/category/gourmet/',
    'https://news.livedoor.com/topics/category/love/',
    'https://news.livedoor.com/topics/category/trend/',
]
# トピックスページだけではなくカテゴリページにも追加
category_ids = [4, 1, 3, 44, 42, 2, 12, 31, 29, 201, 210, 10, 49, 214, 217, 52]
category_urls += ["https://news.livedoor.com/article/category/{}/".format(
    idx) for idx in category_ids]

article_bin_path = './data/article_infos.bin'
article_bin_path_slack = "./data/article_infos_slack.bin"
preprocessd_csv_path = "./data/article.csv"
dataset_path = "./data/datasets.bin"

# const value
START_PAGER_IDX = 1
END_PAGER_IDX = 30
IS_SELENIUM = True

# slack
POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
FILE_UPLOAD_URL = "https://slack.com/api/files.upload"
TOKEN = os.environ.get("SLACK_BOT_TOKEN")
USER_NAME = os.getlogin()
CHANNEL = "scraping"
