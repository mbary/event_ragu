import os
from pathlib import Path
import asyncio
EVENT_DIR_PATH=Path("./events")

event_files=[Path(file).resolve() for file in os.listdir(EVENT_DIR_PATH) if file.endswith('.md')]
a=[x for x in EVENT_DIR_PATH.iterdir() ]
len(event_files)
event_files[0]

async def get_urls(path: Path) -> list[str]:
    """
    Extract URLs from a file.
    """
    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return content[0].split(" ")[-1].lstrip().strip("/\n")

async def main():
    """
    Main function to run the URL extraction.
    """
    tasks = [get_urls(file) for file in a]
    return await asyncio.gather(*tasks)

result = asyncio.run(main())
print(result)
get_urls(a[0])
