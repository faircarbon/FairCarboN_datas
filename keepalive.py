#python -m pip install playwright
#python -m playwright install chromium

#conda deactivate
#conda activate FairCarboN_env
#python -c "import sys; print(sys.executable)" => voir quel python est utilisé

import asyncio
from playwright.async_api import async_playwright

URL = "https://faircarbon-datas.streamlit.app/"

async def wake_app():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print(f"Visite de : {URL}")
        await page.goto(URL, wait_until="networkidle")

        # Cherche le bouton de réveil
        try:
            button = await page.wait_for_selector('text="Yes, get this app back up!"', timeout=5000)
            if button:
                print("L'app est endormie. Réveil en cours…")
                await button.click()
                await page.wait_for_timeout(5000)
        except:
            print("L'app est déjà réveillée.")

        await browser.close()

asyncio.run(wake_app())