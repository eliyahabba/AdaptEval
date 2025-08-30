import asyncio
import pandas as pd
from playwright.async_api import async_playwright


async def scrape_helm_data():
    """
    Scrape HELM data using direct URL navigation with proper page refresh
    to prevent caching issues.
    """
    all_data = []

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            # Disable cache to ensure we get fresh data
            bypass_csp=True,
            viewport={"width": 1280, "height": 720}
        )
        page = await context.new_page()

        # Configure the page to not use cache
        await page.route("**/*", lambda route: route.continue_(
            headers={
                **route.request.headers,
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        ))

        page_number = 1
        max_pages = 30  # Safety limit
        consecutive_empty_pages = 0
        max_empty_pages = 3  # Stop after 3 consecutive pages with no data

        print("Starting to scrape HELM data...")

        while page_number <= max_pages and consecutive_empty_pages < max_empty_pages:
            print(f"Processing page {page_number}...")

            # Full URL including hash
            url = f"https://crfm.stanford.edu/helm/lite/latest/#/runs?page={page_number}"

            try:
                # Navigate to the page and force a complete page load
                # The reload=True option ensures the page is fully reloaded
                await page.goto(url, wait_until="networkidle")

                # After navigation, force a reload to ensure we get fresh data
                print(f"Reloading page {page_number} to ensure fresh data...")
                await page.reload(wait_until="networkidle")

                # Additional wait to ensure JS has executed
                await page.wait_for_timeout(2000)

                # Verify we're on the correct page by checking URL
                current_url = page.url
                print(f"Current URL after reload: {current_url}")

                # Try to get table data
                table_rows = await page.query_selector_all("table tbody tr")

                if not table_rows or len(table_rows) == 0:
                    print(f"No table rows found on page {page_number}")
                    consecutive_empty_pages += 1
                    page_number += 1
                    continue

                # Check if we're on the right page by examining pagination
                pagination_active = await page.evaluate("""() => {
                    const pagination = document.querySelector('.pagination');
                    if (!pagination) return null;

                    const activeItem = pagination.querySelector('li.active');
                    if (!activeItem) return null;

                    return activeItem.textContent.trim();
                }""")

                if pagination_active:
                    print(f"Pagination shows we're on page: {pagination_active}")

                # Extract data from the table
                table_data = await page.evaluate("""() => {
                    const rows = Array.from(document.querySelectorAll('table tbody tr'));
                    return rows.map(row => {
                        const cells = Array.from(row.querySelectorAll('td'));
                        if (cells.length >= 5) {
                            return {
                                'Run': cells[0].textContent.trim(),
                                'Model': cells[1].textContent.trim(),
                                'Groups': cells[2].textContent.trim(),
                                'Adapter method': cells[3].textContent.trim(),
                                'Subject / Task': cells[4].textContent.trim() || '-'
                            };
                        }
                        return null;
                    }).filter(item => item !== null);
                }""")

                if table_data and len(table_data) > 0:
                    # Reset empty page counter since we found data
                    consecutive_empty_pages = 0

                    # Log first row to show we have data
                    print(f"First row on page {page_number}: {table_data[0]['Run']}")

                    # Add data to our collection
                    all_data.extend(table_data)
                    print(f"Collected {len(table_data)} rows from page {page_number}")
                else:
                    print(f"No data extracted from page {page_number}")
                    consecutive_empty_pages += 1

                # Take a screenshot for verification (optional)
                await page.screenshot(path=f"page_{page_number}.png")

                # Move to next page
                page_number += 1

            except Exception as e:
                print(f"Error on page {page_number}: {str(e)}")
                consecutive_empty_pages += 1
                page_number += 1

        # Close browser
        await browser.close()

    return all_data


async def main():
    data = await scrape_helm_data()

    if data:
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        csv_filename = "helm_data.csv"
        df.to_csv(csv_filename, index=True)
        print(f"Data saved to {csv_filename}. Total records: {len(data)}")

        # Print first and last few rows to verify
        print("\nFirst few rows of collected data:")
        print(df.head(3))
        print("\nLast few rows of collected data:")
        print(df.tail(3))

        # Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"Number of duplicates: {duplicates}")
    else:
        print("No data was collected.")


if __name__ == "__main__":
    asyncio.run(main())