import unittest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from spider import Spider, CHROME_PATH, clean_data
from pprint import pprint

DIC = {'Age': 'Everyone',
       'Description': 'Welcome aboard, Captain!\n'
       '\n'
       'The crew is just waiting for your order to set sail, all '
       'ready to go off in search of countless treasures and '
       'spectacular adventures!\n'
       'Look for ancient maps you can use to beat fascinating match 3 '
       'levels and find unheard-of riches.\n'
       '\n'
       "Here's what you can expect in Pirate Treasures:\n"
       '\n'
       '- Gorgeous graphics\n'
       '- A great soundtrack and juicy sound effects\n'
       '- Your friends, who will try to beat you to the treasures\n'
       '- The match 3 gameplay you love\n'
       '- Thousands and thousands of engaging levels\n'
       '- A daring band of sea dogs as your crew\n'
       '- And TREASURES (obviously!)\n'
       '\n'
       'Cast off! Set sail!',
       'Developer': 'TAPCLAP',
       'Inapp_Products': '$0.99 - $25.99 per item',
       'Installs': '10,000,000+',
       'Interactive': 'Digital Purchases',
       'Name': 'Pirate Treasures - Gems Puzzle',
       'Price': 'Install',
       'Rating': '4.5',
       'Rating_1': '33,143',
       'Rating_2': '13,435',
       'Rating_3': '27,502',
       'Rating_4': '74,170',
       'Rating_5': '431,522',
       'Rating_Total': '579,772',
       'Requires_Android': 'Varies with device',
       'Size': 'Varies with device',
       'Updated': 'April 19, 2018'}

VEC = ['puzzle',
       'com.orangeapps.piratetreasure',
       'Pirate Treasures - Gems Puzzle',
       '1526452848',
       '',
       '10000000',
       '',
       'Everyone',
       '0.99',
       '25.99',
       'TAPCLAP',
       '4.5',
       '579772',
       'Digital Purchases',
       '431522',
       '74170',
       '27502',
       '13435',
       '33143',
       '0',
       '1',
       'Welcome aboard Captain!  The crew is just waiting for your order to set sail '
       'all ready to go off in search of countless treasures and spectacular '
       'adventures! Look for ancient maps you can use to beat fascinating match 3 '
       "levels and find unheard-of riches.  Here's what you can expect in Pirate "
       'Treasures:  - Gorgeous graphics - A great soundtrack and juicy sound effects '
       '- Your friends who will try to beat you to the treasures - The match 3 '
       'gameplay you love - Thousands and thousands of engaging levels - A daring '
       'band of sea dogs as your crew - And TREASURES (obviously!)  Cast off! Set '
       'sail!']


class TestSpider(unittest.TestCase):

    def setUp(self):
        options = Options()
        options.add_experimental_option(
            'prefs', {'intl.accept_languages': 'en,en_US'})
        self.driver = webdriver.Chrome(
            executable_path=CHROME_PATH, chrome_options=options)
        self.spider = Spider(self.driver)

    def tearDown(self):
        self.driver.quit()

    def test_parse_page(self):
        self.spider.get_page_by_package('com.orangeapps.piratetreasure')
        dic = self.spider.parse_current_page()
        # self.assertEqual(dic, DIC)
        pprint(dic)
        vec = clean_data(dic, category='puzzle',
                         package='com.orangeapps.piratetreasure')
        # self.assertEqual(vec, VEC)
        pprint(vec)

if __name__ == '__main__':
    unittest.main()
