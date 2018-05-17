import unittest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from spider import Spider, PackageInfoWriter, CHROME_PATH, HEADERS
from pprint import pprint


class TestSpiderAndWriter(unittest.TestCase):

    def setUp(self):
        options = Options()
        options.add_experimental_option(
            'prefs', {'intl.accept_languages': 'en,en_US'})
        self.driver = webdriver.Chrome(
            executable_path=CHROME_PATH, options=options)
        self.spider = Spider(self.driver)
        self.writer = PackageInfoWriter(-1, -1, -1, -1)
        self.target_package = 'com.tencent.ig'

    def tearDown(self):
        self.driver.quit()

    def test(self):
        self.spider.get_page_by_package('com.tencent.ig')
        dic = self.spider.parse_current_page()
        new_dic = dic.copy()
        pprint(dic)

        vec = self.writer.clean_data(dic, category='Action',
                                package=self.target_package)
        pprint(vec)
        self.assertEqual(len(vec), self.writer.valid_length(vec))
        self.assertEqual(len(vec), len(HEADERS['old']))
        

        vec = self.writer.vectorize_dic(new_dic, category='Action',
                               package=self.target_package)

        pprint(vec)
        self.assertEqual(len(vec), self.writer.valid_length(vec))
        self.assertEqual(len(vec), len(HEADERS['new']))
        



if __name__ == '__main__':
    unittest.main()
