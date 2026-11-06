"""Visit the public demo, wake it if needed, and require a usable Streamlit UI."""
import os
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.getenv('STREAMLIT_URL', 'https://steel-defect-detection-52wcl85z79gio5qv82slyd.streamlit.app/')
ARTIFACTS = Path('keepalive-artifacts')


def visit(page, timeout=240):
    page.goto(URL, wait_until='domcontentloaded', timeout=60000)
    deadline = time.monotonic() + timeout
    clicked = False
    while time.monotonic() < deadline:
        # A 200 response can be the sleeping page or an empty JavaScript shell.
        for frame in page.frames:
            wake = frame.get_by_role('button', name=re.compile(r'Yes, get this app back up|Wake.*app', re.I))
            if not clicked and wake.count() and wake.first.is_visible():
                wake.first.click(timeout=10000)
                clicked = True
                print('Clicked the public wake-up button.', flush=True)
        errors = page.locator('[data-testid="stException"]')
        if errors.count() and errors.first.is_visible():
            raise RuntimeError(errors.first.inner_text())
        heading = page.get_by_role('heading', name=re.compile('工业钢材表面缺陷检测系统'))
        uploader = page.locator('[data-testid="stFileUploader"] input[type="file"]')
        if heading.count() and heading.first.is_visible() and uploader.count():
            # Leave the real browser session connected for activity to register.
            page.wait_for_timeout(20000)
            if errors.count() and errors.first.is_visible():
                raise RuntimeError(errors.first.inner_text())
            print('PASS: app heading and image uploader loaded.', flush=True)
            return
        page.wait_for_timeout(2000)
    raise RuntimeError('App did not render its heading and image uploader within the time limit.')


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'channel': os.environ['BROWSER_CHANNEL']} if os.getenv('BROWSER_CHANNEL') else {}))
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        logs = []
        page.on('pageerror', lambda error: logs.append(str(error)))
        try:
            for attempt in range(1, 4):
                try:
                    print(f'Browser visit {attempt}/3: {URL}', flush=True)
                    visit(page)
                    page.screenshot(path=str(ARTIFACTS / 'success.png'), full_page=True)
                    return
                except Exception as exc:
                    print(f'Attempt {attempt} failed: {exc}', flush=True)
                    page.screenshot(path=str(ARTIFACTS / f'failure-{attempt}.png'), full_page=True)
                    (ARTIFACTS / f'failure-{attempt}.txt').write_text(page.locator('body').inner_text() + '\n' + '\n'.join(logs), encoding='utf-8')
                    if attempt == 3:
                        raise
                    page.wait_for_timeout(15000)
        finally:
            browser.close()


if __name__ == '__main__':
    main()
