import utils
import config
slacker = utils.Slacker()

slacker.post_file(file_path="./config.py")
# slacker.post_message("hello world")