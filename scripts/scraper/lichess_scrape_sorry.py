import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging

class AnalysisSpider(scrapy.Spider):
    name = "analysis_spider"
    start_urls = ['https://lichess.org/training/hangingPiece']  # Replace this URL with the actual page you're interested in

    def parse(self, response):
        # Find the button using XPath and perform a click action if it's a link directly
        button_link = response.xpath('//a[contains(@class, "button") and contains(text(), "View the solution")]/@href').get()
        if button_link:
            yield scrapy.Request(url=response.urljoin(button_link), callback=self.parse_after_click)
        else:
            self.logger.error("Button 'View the solution' not found")


    def parse_after_click(self, response):
        # Search for a string that starts with "analysis/" and extract what follows
        analysis_url = response.xpath('//*[contains(text(), "analysis/")]/text()').get()
        if analysis_url:
            analysis_info = analysis_url.split("analysis/")[-1]  # Extract the part after "analysis/"
            with open('hanging_piece.txt', 'a') as file:
                file.write(analysis_info + '\n')
            yield {
                'analysis_info': analysis_info
                }
        else:
            self.logger.error("Analysis information not found")

# Configure logging
configure_logging()

# Create a CrawlerProcess with settings
process = CrawlerProcess(settings={
    'USER_AGENT': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)'
})

# Add the spider to the process
process.crawl(AnalysisSpider)

# Start the crawling process
process.start()
