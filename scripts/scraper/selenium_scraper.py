from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup Chrome with Selenium WebDriver
options = Options()
options.headless = False  # Set to True if you don't need a GUI
service = Service('/Users/jack/Downloads/chrome-mac-arm64/Google-Chrome-for-Testing.app')  # Update the path to your WebDriver
driver = webdriver.Chrome(service=service, options=options)

# Navigate to the webpage
url = 'https://lichess.org/training/hangingPiece'
driver.get(url)

try:
    # Wait for the button to be clickable
    wait = WebDriverWait(driver, 10)  # Adjust timeout as necessary
    button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "View the solution")))

    # Click the button
    button.click()

    # Optionally, handle the content loaded after the click or extract other data
    # ...

finally:
    # Close the browser after completing the tasks
    driver.quit()