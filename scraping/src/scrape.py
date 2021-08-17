from bs4 import BeautifulSoup

import config
import utils

if __name__ == "__main__":
    driver = utils.get_cli_webdriver(config.IS_SELENIUM)
    article_infos = utils.load_article_bin()
    slacker = utils.Slacker()
    try:
        for category_url in config.category_urls[::-1][0:1]:
            print("カテゴリ　{} からスクレイピング開始".format(category_url))
            slacker.post_message("カテゴリ　{} からスクレイピング開始".format(category_url))
            # 初期変数
            pager_idx = config.START_PAGER_IDX
            # 既にスクレイピング済みのサイトのURL
            scraped_page_urls = set([article[0] for article in article_infos])
            
            # ex. https://news.livedoor.com/topics/category/main/?p=1 にアクセスして解析
            category_pager_url = "{0}?p={1}".format(category_url, pager_idx)
            driver.get(category_pager_url)
            print("{}のサイトから1つずつページをスクレイピング".format(category_pager_url))
            utils.rand_sleep()
            # 大元のソープ
            soup = BeautifulSoup(driver.page_source, "html.parser")
            # 次のページへがなくなるまで繰り返す
            while True:
            # 実際にデータを収集するページのリンクを抜き取る
                for ele in soup.find("div", attrs={"class": "mainBody"}).find("ul", attrs={"class": "articleList"}).find_all("li"):
                    # リンクを修正する
                    page_url = ele.find("a").get("href").replace("topics", "article")
                    # すでにスクレイピング済みだったら
                    if page_url in scraped_page_urls:
                        print("{}は既にスクレイピング済です".format(page_url))
                        continue
                    # ex. https://news.livedoor.com/article/detail/20558207/にアクセスする
                    print("{}をスクレピング開始".format(page_url))
                    driver.get(page_url)
                    utils.rand_sleep()
                    
                    # スクレピング対象の記事
                    article_soup = BeautifulSoup(driver.page_source, "html.parser")
                    # データ習得
                    try:
                        article_category = "<SEP>".join([c.text for c in article_soup.find("nav", attrs={"class": "breadcrumbs"}).find_all("span", attrs={"itemprop": "name"})])
                    except:
                        article_category = "None Category"
                    try:
                        article_keyword = "<SEP>".join([k.text for k in article_soup.find("ul", attrs={"class": "articleHeadKeyword"}).find_all("a")])
                    except:
                        article_keyword = "None Keyword"
                    try:
                        article_date = article_soup.find("time", attrs={"class": "articleDate"}).text
                    except:
                        article_date = "None Date"
                    try:
                        article_vender = article_soup.find("p", attrs={"class": "articleVender"}).find("span", attrs={"itemprop": "name"}).text.strip(" ").strip("\n")
                    except:
                        article_vender = "None Vender"
                    try:
                        article_tweet_counts = article_soup.find("ul", attrs={"class": "socialBtn"}).find("div", attrs={"class": "tweet_counts"}).text
                    except:
                        article_tweet_counts = 0
                    try:
                        article_title = article_soup.find("h1", attrs={"class": "articleTtl"}).text
                    except:
                        article_title = "None Title"
                    try:
                        article_text = article_soup.find("span", attrs={"itemprop": "articleBody"}).text
                    except:
                        article_text = "None Text"
                    # データを一旦listに保存
                    article_infos.append([page_url, article_date, article_title, article_text, article_category, article_keyword, article_tweet_counts, article_vender])
                    print("{}の抽出完了".format(page_url))
                    print("記事のタイトル: {0}, 記事の本文一部: {1}".format(article_title, article_text[0:10]))
                
                # ?p=〇ごとにデータ保存
                print("記事数{}をpickleに保存します".format(len(article_infos)))
                utils.save_article_bin(article_infos)
                    
                # 30ページスクレイピングしたら次のカテゴリへ
                if pager_idx >= config.END_PAGER_IDX:
                    print("次のカテゴリへ進みます")
                    break
                    
                # 次のページを進むのボタンがあれば
                if soup.find("ul", attrs={"class": "pager"}).find("li", attrs={"class": "next"}):
                    pager_idx += 1
                    category_pager_url = "{0}?p={1}".format(category_url, pager_idx)
                    print("{0}ページ目の{1}に進みます".format(pager_idx, category_pager_url))
                    driver.get(category_pager_url)
                    utils.rand_sleep()
                    
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                # 最後のページのためループを抜ける
                else:
                    print("次のカテゴリへ進みます")
                    break
        slacker.post_file(config.article_bin_path)
    except Exception as e:
        slacker.post_message(str(e))