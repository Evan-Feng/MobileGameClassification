from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import multiprocessing as mp
import json
import csv
import time
import argparse

N_PROCESS = 8
PACKAGES_PER_EPOCH = 40
CHROME_PATH = 'Z:/chromedriver.exe'
CATEGORYIES = ['Action', 'Adventure', 'Arcade', 'Board', 'Card',
               'Casino', 'Casual', 'Educational', 'Music', 'Puzzle',
               'Racing', 'Role_Playing', 'Simulation', 'Sports',
               'Strategy', 'Trivia', 'Word']
GOOGLE_PLAY_BASE_URL = 'https://play.google.com/store/apps/'
XPATHS = {
    'general': {
        'Name': "//h1[@itemprop = 'name']/span",
        'Price': "//button[@aria-label and span[span]]",
        'Updated': "//div[div='Updated']/span/div/span",
        'Size': "//div[div='Size']/span/div/span",
        'Installs': "//div[div='Installs']/span/div/span",
        'Requires_Android': "//div[div='Requires Android']/span/div/span",
        'Age': "//div[div='Content Rating']/span/div/span/div",
        'Inapp_Products': "//div[div='In-app Products']/span/div/span",
        'Developer': "//div[div='Offered By']/span/div/span",
        'Description': "//content/div[@jsname='sngebd']",
        'Rating': "//div[@class='BHMmbe']",
        'Rating_Total': "//span[@aria-label]",
        'Content_Feature': "//div[div='Content Rating']/span/div/span/div[2]",
        'Version': "//div[div='Current Version']/span/div/span",
    },
    'rating': {
        'Rating_5': "//div[span = '5']/span[@title]",
        'Rating_4': "//div[span = '4']/span[@title]",
        'Rating_3': "//div[span = '3']/span[@title]",
        'Rating_2': "//div[span = '2']/span[@title]",
        'Rating_1': "//div[span = '1']/span[@title]",
    },
    'permission': {
        'button': "//div[div = 'Permissions']/span/div/span/div/a",
        'each_item': "//li[@class = 'NLTG4']/span",
    },
}
HEADERS = {
    'full': ['Category', 'Package', 'Name', 'Updated', 'Size',
             'Installs', 'Requires_Android', 'Age', 'Developer', 'Rating',
             'Rating_Total', 'Rating_5', 'Rating_4', 'Rating_3', 'Rating_2',
             'Rating_1', 'Price', 'Description',
             'Content_Feature', 'Permission', 'Inapp_Products', 'Version'],
    'trivial': ['Inapp_Products'],
}


class Spider:

    def __init__(self, driver):
        self.driver = driver
        time.sleep(5)

    def get_page_by_package(self, package):
        self.driver.get(GOOGLE_PLAY_BASE_URL + 'details?id=' + package)
        time.sleep(3)

    def parse_current_page(self):
        parsed_dic = {}
        for key, xpath in XPATHS['general'].items():
            try:
                parsed_dic[key] = self.driver.find_element_by_xpath(xpath).text
            except Exception as e:
                pass
        for key, xpath in XPATHS['rating'].items():
            try:
                parsed_dic[key] = self.driver.find_element_by_xpath(
                    xpath).get_attribute('title')
            except Exception as e:
                pass
        try:
            self.driver.find_element_by_xpath(
                XPATHS['permission']['button']).click()
            time.sleep(2)
            parsed_dic['Permission'] = ';'.join(
                [elem.text for elem in self.driver.find_elements_by_xpath(XPATHS['permission']['each_item'])])
        except Exception as e:
            pass
        return parsed_dic


class PackageInfoWriter:

    def __init__(self, csv_path, period, strict_mode):
        self.buffer, self.count = [], 0
        self.path, self.period, self.strict = csv_path, period, strict_mode

    def write(self):
        with open(self.path, 'a', encoding='utf-8', newline='') as fout:
            csvout = csv.writer(fout)
            csvout.writerows(self.buffer)
        self.buffer = []

    def process_dic(self, dic, category, package):

        def vectorize_dic(dic, *, category, package):
            for key in HEADERS['trivial']:
                dic[key] = dic.get(key, '???')
            dic['Category'], dic['Package'] = category, package
            return [dic.get(key, '').replace('\n', ' ') for key in HEADERS['full']]

        vec = vectorize_dic(dic, category=category, package=package)
        if len(dic) < 10 or self.strict and '' in vec:
            print('[%s] pakcage %s missing property %d' %
                  (category, package, vec.index('')))
            return False

        self.buffer.append(vec)
        self.count += 1
        if self.count % self.period == 0:
            self.write()
        return True

    def close(self):
        self.write()


def scrape_category(category, args, maxnum):
    with open(args.infile, 'r') as fin:
        dic = json.load(fin)

    with open('raw/%s.csv' % category, 'r', encoding='utf-8') as fin:
        csvin = csv.DictReader(fin)
        visited = {row['Package'] for row in csvin}

    with open('log/%s_visited.json' % category, 'r') as fin:
        visited |= set(json.load(fin))

    target_packages = set(dic[category]) - visited
    count = 0

    print('Already visited %d packages in category %s' %
          (len(visited), category))

    writer = PackageInfoWriter('raw/%s.csv' % category, 5, args.strict)
    options = Options()
    options.add_experimental_option(
        'prefs', {'intl.accept_languages': 'en,en_US'})
    if args.headless:
        options.add_argument('--headless')
    options.add_argument('--log-level=3')
    driver = webdriver.Chrome(executable_path=CHROME_PATH, options=options)

    try:
        spider = Spider(driver)
        while target_packages and count < maxnum:
            package = target_packages.pop()
            if package in visited:
                continue
            else:
                visited.add(package)

            spider.get_page_by_package(package)
            parsed_dic = spider.parse_current_page()
            if writer.process_dic(parsed_dic, category, package):
                print('[%s] package %s scraped' % (category, package))
                count += 1
        else:
            print('All packages in category %s has been scraped' % category)
    finally:
        with open('log/%s_visited.json' % category, 'w') as fout:
            json.dump(list(visited), fout, indent=4)
        print('Category %s exit' % category)
        writer.close()
        driver.quit()
    return count if target_packages else -1 


def scrape_multiple_categories(categories, args, maxnum):
    print('Process started. Scraping %s' % categories)
    while categories and maxnum > 0:
        cate = categories.pop(0)
        try:
            res = scrape_category(cate, args, maxnum=PACKAGES_PER_EPOCH)
            if res is None or res >= 0:
                maxnum -= res
                categories.append(cate)
        except Exception as e:
            print(e)
    print('Process terminating...')


def main(args):
    div, mod = divmod(len(CATEGORYIES), N_PROCESS)
    ncate = div + 1 if mod else div
    for i in range(0, div * (N_PROCESS - mod), div):
        mp.Process(target=scrape_multiple_categories, args=(
            CATEGORYIES[i:i + div], args, args.n // N_PROCESS)).start()
    for i in range(div * (N_PROCESS - mod), len(CATEGORYIES), div + 1):
        p = mp.Process(target=scrape_multiple_categories, args=(
            CATEGORYIES[i:i + div + 1], args, args.n // N_PROCESS)).start()


if __name__ == '__main__':
    description = 'This is a program that automatically scrapes Google Play ' + \
        'packages info in certain categories. ' + \
        'Specify the categories by using the following indices:\n \n'
    for i in range(5, len(CATEGORYIES) + 1, 5):
        description += '%d - %d: ' % (i - 5, i - 1) + \
            ', '.join(CATEGORYIES[i - 5:i]) + '\n'
    else:
        if len(CATEGORYIES) % 5 != 0:
            description += '%d - %d: ' % (i, len(CATEGORYIES) - 1) + \
                ', '.join(CATEGORYIES[i:len(CATEGORYIES)])

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile',
                        help='specify the json file that contains package names')
    parser.add_argument('-n',
                        type=int,
                        default=20000,
                        help="the number of packages to scrape (defalt: 20000)")
    parser.add_argument('--headless',
                        action='store_true',
                        help="use the headless version of Chrome")
    parser.add_argument('--strict',
                        action='store_true',
                        help="skip the package if at least one property is missing")
    args = parser.parse_args()
    main(args)
