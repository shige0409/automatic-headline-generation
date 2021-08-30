import utils
import config

if __name__ == "__main__":
    slacker = utils.Slacker()
    slacker.post_file(file_path=config.article_bin_path)