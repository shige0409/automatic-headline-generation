# スクレイピングTips
1. URL/robots.txtにスクレイピングのルールが記載 => 法的拘束力はないがマナーは守ろう
2. <meta name="robots" content="〇〇"> => ↑と同様
3. エージェント設定 => これがないと弾かれるサイトもある
```python
http://httpbin.org/useragent => "Google Chrome/90.1..."
request.get(url, headers={"UserAgent": "↑の値"})
```
4. リファラ(アクセス元)設定　=> これがないとサイトの挙動が変わる場合も
```python
request.get(url, refer="アクセス元")
```

5. プロキシサーバー設定