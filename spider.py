from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from multiprocessing import Pool
import json
import csv
import time
import argparse

LOCAL_WRITE_PERIOUD = 5

CHROME_PATH = 'Z:/chromedriver.exe'

CATEGORYIES = ['Action', 'Adventure', 'Arcade', 'Board', 'Card',
               'Casino', 'Casual', 'Educational', 'Music', 'Puzzle',
               'Racing', 'Role_Playing', 'Simulation', 'Sports',
               'Strategy', 'Trivia', 'Word']

GOOGLE_PLAY_BASE_URL = 'https://play.google.com/store/apps/'


XPATHS_GENERAL = {
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
    'Interactive': "//div[div='Interactive Elements']/span/div/span"
}

XPATHS_RATING = {
    'Rating_5': "//div[span = '5']/span[@title]",
    'Rating_4': "//div[span = '4']/span[@title]",
    'Rating_3': "//div[span = '3']/span[@title]",
    'Rating_2': "//div[span = '2']/span[@title]",
    'Rating_1': "//div[span = '1']/span[@title]",
}

HEADERS = ['Category', 'Package', 'Name', 'Updated', 'Size',
           'Installs', 'Requires_Android', 'Age', 'Inapp_Lower_Bound', 'Inapp_Upper_Bound',
           'Developer', 'Rating', 'Rating_Total', 'Interactive', 'Rating_5',
           'Rating_4', 'Rating_3', 'Rating_2', 'Rating_1', 'Price',
           'Free', 'Description']

KEY_TO_INDEX = {key: i for i, key in enumerate(HEADERS)}


class Spider:

    def __init__(self, driver):
        self.driver = driver

    def get_page_by_package(self, package):
        self.driver.get(GOOGLE_PLAY_BASE_URL + 'details?id=' + package)
        time.sleep(2)

    def parse_current_page(self):
        parsed_dic = {}
        for key, xpath in XPATHS_GENERAL.items():
            try:
                parsed_dic[key] = self.driver.find_element_by_xpath(xpath).text
            except Exception as e:
                print('Error occured when parsing general properties', e)
        for key, xpath in XPATHS_RATING.items():
            try:
                parsed_dic[key] = self.driver.find_element_by_xpath(
                    xpath).get_attribute('title')
            except Exception as e:
                print('Error occured when parsing ratings', e)
        return parsed_dic


def clean_data(dic, *, category, package):
    if 'Inapp_Products' in dic:
        tmp = dic['Inapp_Products'].replace(
            ' per item', '').replace(' - ', '-')
        dic['Inapp_Lower_Bound'] = tmp.split('-')[0][1:]
        dic['Inapp_Upper_Bound'] = tmp.split('-')[1][1:]
    else:
        dic['Inapp_Lower_Bound'], dic['Inapp_Upper_Bound'] = '0', '0'

    if 'Updated' in dic:
        tm = time.strptime(dic['Updated'], '%B %d, %Y')
        days = str(int((time.time() - time.mktime(tm)) / 86400))
        dic['Updated'] = days

    if 'Requires_Android' in dic:
        if dic['Requires_Android'].startswith('Varies'):
            dic['Requires_Android'] = ''
        else:
            tmp = dic['Requires_Android'].replace(' and up', '')
            dic['Requires_Android'] = '.'.join(tmp.split('.')[:2])

    if 'Size' in dic:
        dic['Size'] = dic['Size'][
            :-1] if not dic['Size'].startswith('Varies') else ''

    if 'Price' in dic:
        if dic['Price'] == 'Install':
            dic['Price'], dic['Free'] = '0', '1'
        else:
            dic['Price'], dic['Free'] = str(
                float(dic['Price'].split()[0][1:])), '0'

    if 'Installs' in dic:
        dic['Installs'] = dic['Installs'][:-1]

    for key in dic:
        dic[key] = dic[key].replace(',', '').replace('\n', ' ')

    dic['Category'] = category
    dic['Package'] = package

    vector = [dic.get(key, '') for key in HEADERS]
    return vector


def scrape_category(category, *, headless):
    with open('game_packages.json') as fin:
        dic = json.load(fin)

    with open('raw/' + category + '.csv', 'r', encoding='utf-8') as fin:
        csvin = csv.DictReader(fin)
        visited = {row['Package'] for row in csvin}

    print('Already visited %d packages in category %s' %
          (len(visited), category))

    target_packages = set(dic[category]) - visited

    buffer = []

    try:
        count = 0

        options = Options()
        options.add_experimental_option(
            'prefs', {'intl.accept_languages': 'en,en_US'})
        if headless:
            options.add_argument('--headless')
        driver = webdriver.Chrome(executable_path=CHROME_PATH, options=options)
        spider = Spider(driver)
        time.sleep(5)

        while target_packages:
            package = target_packages.pop()
            if package in visited:
                continue
            print(category, count, package)
            spider.get_page_by_package(package)
            parsed = spider.parse_current_page()
            if len(parsed) < 10:
                print('Missing properties:', parsed)
                continue
            buffer.append(clean_data(
                parsed, category=category, package=package))

            if count % LOCAL_WRITE_PERIOUD == 0:
                with open('raw/' + category + '.csv', 'a', encoding='utf-8', newline='') as fout:
                    csvout = csv.writer(fout)
                    csvout.writerows(buffer)
                buffer = []

            visited.add(package)
            count += 1
    except Exception as e:
        print('Error occured when scraping %s' % category, e)
    finally:
        print('Category %s exit' % category)
        driver.quit()


def main(args):
    print(args)
    p = Pool()
    for category_index in args.indices:
        p.apply_async(scrape_category, (CATEGORYIES[
                      category_index],), {'headless': args.headless})
    p.close()
    p.join()


if __name__ == '__main__':
    desc = ''
    for i in range(5, len(CATEGORYIES) + 1, 5):
        desc += '%d - %d: ' % (i - 5, i - 1) + \
            ' '.join(CATEGORYIES[i - 5:i]) + '\n'
    else:
        if len(CATEGORYIES) % 5 != 0:
            desc += '%d - %d: ' % (i, len(CATEGORYIES) - 1) + \
                ' '.join(CATEGORYIES[i:len(CATEGORYIES)])

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('indices',
                        type=int,
                        nargs='+',
                        help='specify the indices of the categories to scrape')
    parser.add_argument('--headless',
                        action='store_true',
                        help="use the headless version of Chrome")
    args = parser.parse_args()
    main(args)
