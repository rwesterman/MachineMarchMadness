# -*- coding: utf-8 -*-
import scrapy
import os
import csv

class BpiSpider(scrapy.Spider):
    name = 'BPI'

    def __init__(self):
        super().__init__()
        # self.logger.setLevel("Warning")
        # self.last_row = {"2008": 0, "2009": 0, "2010": 0, "2011": 0, "2012": 0,
        #                  "2013": 0, "2014": 0, "2015": 0, "2016": 0, "2017": 0, "2018": 0}

    # allowed_domains = ['http://www.espn.com/mens-college-basketball/bpi/_/view/bpi']
    def start_requests(self):
        urls = ['http://www.espn.com/mens-college-basketball/bpi/_/season/2018/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2017/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2016/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2015/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2014/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2013/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2012/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2011/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2010/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2009/view/bpi',
                'http://www.espn.com/mens-college-basketball/bpi/_/season/2008/view/bpi']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        headers = []

        # Extract year from url
        year = response.url.split("/")[-3]
        # Get first three column headers ['RK', 'TEAM', 'CONF']
        headers.extend(response.xpath('//thead/tr/th/text()').extract())
        # Get remaining col headers ['W-L', 'BPI Off', 'BPI Def', 'BPI', '7-Day RK CHG']
        headers.extend(response.xpath('//thead/tr/th/a/text()').extract())
        # Contains all team names on page
        team_names = response.xpath('//tbody/tr/td/a/span/span/text()').extract()
        # Get all data relating to each team. This will need to be split to line up with the team name
        all_data = response.xpath('//tbody/tr/td/text()').extract()

        # Split the lists so each team's data lines up with their name
        team_data = []
        for index in range(0,len(all_data)-5,6):
            team_data.append(all_data[index:index+6])


        # Zip the data together so each tuple is (team_name, team_data)
        team_comp = zip(team_names, team_data)

        self.write_csv(team_comp, headers, year)

        next_page = response.xpath('//ul[@class="pagination"]/li/a'
                                   '[@class = "button-filter arrow-btn right webview-internal "]/@href').extract_first()

        if next_page:
            # Create absolute path to next page
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback = self.parse)


    def write_csv(self, data_tuple, header, year):
        """
        Write data to a csv file
        :param data_tuple: A list of tuples with team-specific data
        :param header: list of strings, specifying data in columns
        :param year: year of BPI data
        """
        # Logic to determine if header row needs to be written
        write_header = False
        if not os.path.exists("ESPN_BPI_{}.csv".format(year)):
            write_header = True

        # create a list of lists with organized team data
        data_list = []
        for index, (team_name, data) in enumerate(data_tuple):
            team_data = []
            # self.logger.info("Index: {}, data: {}".format(index, data))

            # data[1][0] is team rank. This should be first item in each interior list
            team_data.append(data[0])

            # data[0] is team name
            team_data.append(team_name)

            # From here I organize the stats as they are presented
            team_data.append(data[1])
            team_data.append(data[2])
            team_data.append(data[3])
            team_data.append(data[4])
            team_data.append(data[5])

            # Need this outside the loop so it doesn't stack iterations of team_data
            data_list.append(team_data)


        # Write all data to csv file, each row being one team and its data
        with open("ESPN_BPI_{}.csv".format(year), 'a', newline = '') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            for team in data_list:
                writer.writerow(team)





