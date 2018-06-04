import scrapy
import os

class KenpomSpider(scrapy.Spider):
    name = "kenpom"

    def start_requests(self):
        # Get all years from starting year (2002) to most recent (2018)
        year = 2002
        urls = []
        while year < 2019:
            urls.append("https://kenpom.com/index.php?y={}".format(year))
            year += 1

        for url in urls:
            yield scrapy.Request(url=url, callback = self.parse)

    def parse(self, response):
        rank_dict = {}
        # hold the header columns
        header_cols = []
        # extract year of data from url
        year = response.url.split('=')[-1]
        filename = "Kenpom_{}.html".format(year)

        header = response.xpath('//thead/tr[@class="thead2"]')
        left = header.xpath('//th[@class="hard_left"]/a[@href]/text()').extract()

        # Get header columns (only the first row of headers though)
        for x in header.xpath('//tr/th/a/text()'):
            if x in header_cols:
                break
            else:
                header_cols.append(x)

        # # Todo: Create list of xpaths to each column in the table, then iterate through them to get the data similar to below
        # #get xpath to table body
        table_body = response.xpath('//div/table[@id="ratings-table"]/tbody')

        # Gonna have to build these manually it seems. Need to use parent::* to limit to seeded teams and keep order consistent
        # team_seed is the list of parents to team name and seed for all tournament seeded teams
        team_seed = table_body.xpath('//td[@class="next_left"]/span[@class="seed"]/parent::*')



        # rank = table_body.xpath('//tr/td[@class="hard_left"]/text()')
        # # extract full list of ranks as value of 'Rk' from header_cols[0]
        # ranks = rank.extract()
        # rank_dict[header_cols[0]] = ranks

        # col = table_body.xpath('//tr/td[@class="next_left"]/a/text()')


        # Note: shockingly, this works. It only returns the text = 1 though
        # response.xpath('//table[@id="ratings-table"]/tbody/tr/td[@class="hard_left"]/text()="1"')




