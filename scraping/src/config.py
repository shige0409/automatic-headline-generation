# これがないとスクレピングできないサイトのため
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
    "refer": "https://news.livedoor.com/"
}

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
category_urls += ["https://news.livedoor.com/article/category/{}/".format(idx) for idx in category_ids]

# path
article_bin_path = '../data/article_infos.bin'
preprocessd_csv_path = "../data/article.csv"

# const value
START_PAGER_IDX = 1
END_PAGER_IDX = 10
IS_SELENIUM = False