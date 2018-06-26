# -*- coding: utf-8 -*-
import logging

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

class EloWalker():

    def __init__(self, start_url):
        self.driver = webdriver.Firefox(executable_path= "geckodriver.exe")
        self.count_pages = 0
        self.start_url = start_url
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level = logging.INFO)
        self.wait = WebDriverWait(self.driver, 10)

        self.driver.get(self.start_url)

    def get_next_page(self):
        """Find out if there is a next page link"""

        self.logger.info("Finding next page")
        # Find element for next page button, wait until it's clickable to be safe
        try:
            next_page = self.wait.until(EC.element_to_be_clickable(
                (By.LINK_TEXT, "Next page")))
                # (By.XPATH, '//div[@id="content"]/div[@id = "pi"]/div/p/a[contains(text(), "Next")]/@href')))
        except TimeoutException as e:
            # return False if next page does not exist
            next_page = False

        return next_page

    def click_next_page(self, next_page):
        # click on next page link. Might need to scroll to view it first
        next_page.click()

    def get_csv_data(self):
        # Todo: Change from scrapy class to selenium class

        # Hover over the share & more option on the webpage
        share_and_more = self.driver.find_element_by_xpath('//*[@id="all_stats"]/div[1]/div/ul/li[@class="hasmore"]/span')
        hover = ActionChains(self.driver).move_to_element(share_and_more)
        hover.perform()

        # Click the button to get data as csv
        csv_button = self.wait.until(EC.element_to_be_clickable(
            (By.XPATH,'//*[@id="all_stats"]/div[1]/div/ul/li[1]/div/ul/li[4]/button')))

        csv_button.click()

        # Copy the csv data from the page
        page_data = self.driver.find_element_by_xpath('//*[@id="csv_stats"]').text

        # write the page_data to a csv file
        self.write_csv(page_data)
        # self.logger.info("End of data is {}".format(page_data[-50:-1]))

    def write_csv(self, data):
        self.logger.debug("Writing to CSV")
        # Write all data to csv file, each row being one team and its data
        with open("Elo_Data_Winners.csv", 'a') as f:
            f.write(data)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # This link is all games from 2002-2018 listed with the winner first. This should be the most complete dataset possible,
    # but COULD CAUSE ISSUES IF THIS DATA IS USED FOR TRAINING A NEURAL NETWORK.
    # I only intend to determine Elo rankings based on this data so win/lose order shouldn't matter.
    crawl = EloWalker('http://cbbref.com/tiny/AdoIJ')
    next_page = crawl.get_next_page()

    # While there is a next page available...
    while next_page:
        # gather the csv data from the page and write it to .csv file
        crawl.get_csv_data()
        # click on the next page
        crawl.click_next_page(next_page)

        next_page = crawl.get_next_page()

    print(crawl.driver.current_url)

    crawl.driver.close()