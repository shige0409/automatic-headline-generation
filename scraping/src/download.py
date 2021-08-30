import module
import config

if __name__ == "__main__":
    slacker = module.Slacker()
    slacker.post_file(file_path=config.article_bin_path)