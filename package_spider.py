from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import argparse
import time
import json
from spider import CATEGORYIES, CHROME_PATH

PACKAGES_PER_EPOCH = 100
XPATHS = {
    'seemore': "//a[text() = 'See more']",
    'package_item': "//span[@class = 'preview-overlay-container']",
}


def scroll_down(driver, clicks):
    time.sleep(2)
    for _ in range(clicks):
        ActionChains(driver).key_down(Keys.DOWN).perform()
    time.sleep(2)


def scrape_packages_in_category(category, driver):
    packages = set()
    try:
        driver.get(
            'https://play.google.com/store/apps/category/GAME_' + category.upper())
        scroll_down(driver, 50)
        n_buttons = len(driver.find_elements_by_xpath(XPATHS['seemore']))

        for i in range(n_buttons):
            for attempt in range(5):
                driver.get(
                    'https://play.google.com/store/apps/category/GAME_' + category.upper())
                scroll_down(driver, 50)
                try:
                    elem = driver.find_elements_by_xpath(XPATHS['seemore'])[i]
                    elem.click()
                except Exception as e:
                    print('Attempt %d Failed.\n' % attempt, e)
                    continue
                break
            else:
                print('Clikcing see-more button Failed.')
                continue

            scroll_down(driver, 1000)
            for elem in driver.find_elements_by_xpath(XPATHS['package_item']):
                packages.add(elem.get_attribute('data-docid'))
            print(category, len(packages))
    finally:
        packages = list(packages)
        with open('temp.txt', 'w') as fout:
            json.dump(packages, fout, indent=4)
    return packages


def scrape_packages_general(driver, maxnum):
    packages = set()
    try:
        driver.get('https://play.google.com/store/apps/category/GAME')
        while len(packages) < maxnum:
            print(len(packages))
            scroll_down(driver, 50)
            for elem in driver.find_elements_by_xpath(XPATHS['package_item']):
                packages.add(elem.get_attribute('data-docid'))
            try:
                driver.find_element_by_xpath(XPATHS['seemore']).click()
            except Exception as e:
                pkg = packages.pop()
                packages.add(pkg)
                driver.get(
                    'https://play.google.com/store/apps/details?id=' + pkg)
    finally:
        packages = list(packages)
        with open('temp.txt', 'w') as fout:
            json.dump(packages, fout, indent=4)
    return packages


def main(args):
    setup_time = time.time()
    firefox_profile = webdriver.FirefoxProfile()
    firefox_profile.set_preference("intl.accept_languages", 'en-us')
    firefox_profile.update_preferences()
    options = Options()
    if args.headless:
        options.add_argument('--headless')
    driver = webdriver.Firefox(
        executable_path='Z:/geckodriver.exe', firefox_options=options, firefox_profile=firefox_profile)

    result = {} if args.c else []
    try:
        if args.c:
            for cate in CATEGORYIES:
                result[cate] = scrape_packages_in_category(cate, driver)
        else:
            result += scrape_packages_general(driver, args.n)

    finally:
        with open('packages.json', 'w') as fout:
            json.dump(result, fout, indent=4)
        print('package scraping finished in %.2f seconds, quiting...' %
              (time.time() - setup_time))
        driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        action='store_true',
                        help='scrape category-related packages')
    parser.add_argument('-n',
                        type=int,
                        default=2000,
                        help='maximum number of packages to scrape (default: 2000)')
    parser.add_argument('--headless',
                        action='store_true',
                        help="use the headless version of Chrome")
    args = parser.parse_args()
    main(args)
