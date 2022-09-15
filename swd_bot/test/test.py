from swd_bot.test.correctness import test_games_correctness
from swd_bot.thirdparty.sevenee import SeveneeLoader


def main():
    test_games_correctness("../../../7wd/sevenee/", SeveneeLoader)


if __name__ == "__main__":
    main()
