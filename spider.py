from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from multiprocessing import Pool
import json
import csv
import time
import argparse

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

        # supported in the old version BUT NOT supported in the new version
        'Interactive': "//div[div='Interactive Elements']/span/div/span",

        # supported in the new version BUT NOT suppoted in the old version
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

# XPATHS_RATING = {
#     'Rating_5': "//div[span = '5']/span[@title]",
#     'Rating_4': "//div[span = '4']/span[@title]",
#     'Rating_3': "//div[span = '3']/span[@title]",
#     'Rating_2': "//div[span = '2']/span[@title]",
#     'Rating_1': "//div[span = '1']/span[@title]",
# }

HEADERS = {
    # total length: 22
    'old': ['Category', 'Package', 'Name', 'Updated', 'Size',
            'Installs', 'Requires_Android', 'Age', 'Inapp_Lower_Bound', 'Inapp_Upper_Bound',
            'Developer', 'Rating', 'Rating_Total', 'Interactive', 'Rating_5',
            'Rating_4', 'Rating_3', 'Rating_2', 'Rating_1', 'Price',
            'Free', 'Description'],

    # total length: 22
    'new': ['Category', 'Package', 'Name', 'Updated', 'Size',
            'Installs', 'Requires_Android', 'Age', 'Developer', 'Rating',
            'Rating_Total', 'Rating_5', 'Rating_4', 'Rating_3', 'Rating_2',
            'Rating_1', 'Price', 'Description',

            'Content_Feature', 'Permission', 'Inapp_Products', 'Version']
}


class Spider:

    def __init__(self, driver, last_pos=(-1, -1), category=None):
        self.driver = driver
        self.pos1, self.pos2 = last_pos
        if category:
            self.driver.get(GOOGLE_PLAY_BASE_URL + 'category/GAME_' + category.upper())
            self.driver.find_element_by_xpath()
            "//div[@class = 'browse-page']/div[7]//a[@class = 'see-more play-button small id-track-click apps id-responsive-see-more']"


    def get_page_by_package(self, package):
        self.driver.get(GOOGLE_PLAY_BASE_URL + 'details?id=' + package)
        time.sleep(3)

    def get_category(self, category):
        self.last = 0
        target_xpath = "//div[@class = 'id-card-list card-list two-cards']" +\
            "/div[%d]//span[@class = 'preview-overlay-container']" % self.pos2
        self.pos2 += 1
        try:
            self.dirver.find_element_by_xpath(target_xpath)
        except Exception:
            self.driver.get(GOOGLE_PLAY_BASE_URL + 'category/GAME_' + category.upper())
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
            time.sleep(0.5)
            parsed_dic['Permission'] = ';'.join(
                [elem.text for elem in self.driver.find_elements_by_xpath(XPATHS['permission']['each_item'])])
        except Exception as e:
            pass
        return parsed_dic


class PackageInfoWriter:

    def __init__(self, use_old_version, csv_path, period, strict_mode):
        self.buffer, self.count = [], 0
        self.old, self.path = use_old_version, csv_path
        self.period, self.strict = period, strict_mode

    def process_package_dic(self, dic, category, package):
        if args.old:
            vec = clean_data(dic, category=category, package=package)
        else:
            vec = vectorize_dic(dic, category=category, package=package)

        if len(dic) < 10 or self.strict and self.old and \
                self.valid_length(vec) < len(HEADERS['old']) or self.strict and \
                not self.old and self.valid_length(vec) < len(HEADERS['new']):
            print('Missing properties:', len(dic))
            return False

        self.buffer.append(vec)
        self.count += 1
        if self.count % self.period == 0:
            with open(self.path, 'a', encoding='utf-8', newline='') as fout:
                csvout = csv.writer(fout)
                csvout.writerows(self.buffer)
            self.buffer = []

    def vectorize_dic(self, dic, *, category, package):
        dic['Category'], dic['Package'] = category, package
        return [dic.get(key, '') for key in HEADERS['new']]

    def valid_length(self, vec):
        return len([0 for v in vec if v != ''])

    def clean_data(self, dic, *, category, package):
        if 'Inapp_Products' in dic:
            tmp = dic['Inapp_Products'].replace(
                ' per item', '').replace(' - ', '-')
            try:
                dic['Inapp_Lower_Bound'] = tmp.split('-')[0][1:]
                dic['Inapp_Upper_Bound'] = tmp.split('-')[1][1:]
            except Exception as e:
                dic['Inapp_Lower_Bound'], dic['Inapp_Upper_Bound'] = '0', '0'
                print('Inapp_Products Error', e)

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

        return [dic.get(key, '') for key in HEADERS['old']]


def scrape_category(category, args):
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
        options = Options()
        options.add_experimental_option(
            'prefs', {'intl.accept_languages': 'en,en_US'})
        if args.headless:
            options.add_argument('--headless')
        driver = webdriver.Chrome(executable_path=CHROME_PATH, options=options)
        spider = Spider(driver)
        writer = PackageInfoWriter(
            args.old, 'raw/' + category + '.csv', 5, args.strict)
        time.sleep(5)

        while target_packages:
            package = target_packages.pop()
            if package in visited:
                continue
            else:
                visited.add(package)
            spider.get_page_by_package(package)
            parsed_dic = spider.parse_current_page()
            writer.process_package_dic(parsed_dic)
        else:
            print('All packages in category %s has been scraped' % category)
    except Exception as e:
        print('Error occured when scraping %s:' % category, e)
        raise
    finally:
        print('Category %s exit' % category)
        driver.quit()


def main(args):
    p = Pool()
    for category_index in set(args.indices):
        p.apply_async(scrape_category, (CATEGORYIES[category_index], args))
    p.close()
    p.join()


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
    parser.add_argument('indices',
                        type=int,
                        nargs='+',
                        help='specify the indices of the categories to scrape')
    parser.add_argument('--headless',
                        action='store_true',
                        help="use the headless version of Chrome")
    parser.add_argument('--strict',
                        action='store_true',
                        help="skip the package if at least one property is missing")
    args = parser.parse_args()
    main(args)
