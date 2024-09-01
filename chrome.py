from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def setup_driver():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    
    # Set up the Chrome driver (make sure you have chromedriver installed and in PATH)
    service = Service('path/to/chromedriver')  # Replace with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def open_new_tab(driver, url):
    # Open a new tab
    driver.execute_script("window.open('');")
    # Switch to the new tab (it's always the last one)
    driver.switch_to.window(driver.window_handles[-1])
    # Navigate to the specified URL
    driver.get(url)

def switch_to_tab(driver, index):
    # Switch to the tab at the specified index
    driver.switch_to.window(driver.window_handles[index])

def close_current_tab(driver):
    # Close the current tab
    driver.close()
    # Switch to the last tab (to ensure we're on an active tab)
    driver.switch_to.window(driver.window_handles[-1])

def main():
    driver = setup_driver()

    # Open Google in the first tab
    driver.get("https://www.google.com")

    # Open a new tab with YouTube
    open_new_tab(driver, "https://www.youtube.com")

    # Wait for 2 seconds
    time.sleep(2)

    # Switch back to the first tab (Google)
    switch_to_tab(driver, 0)

    # Find the search box and enter a query
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.send_keys("Python programming")
    search_box.send_keys(Keys.RETURN)

    # Wait for 2 seconds
    time.sleep(2)

    # Close the current tab (Google search results)
    close_current_tab(driver)

    # The active tab is now YouTube

    # Wait for 5 seconds before closing the browser
    time.sleep(5)

    # Close the browser
    driver.quit()

if __name__ == "__main__":
    main()