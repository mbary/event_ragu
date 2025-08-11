def main():
    print("Hello from my-ai-agent!")


if __name__ == "__main__":
    main()

from brave import Brave

from dotenv import load_dotenv
load_dotenv()
import os
os.environ.get("BRAVE_API_KEY")



brave = Brave()

results = brave.search("What concerts are in warsaw this weekend?", count=20, safesearch='off',#raw=True,
                        freshness='pw', extra_snippets=True)

from pprint import pprint
pprint(results)
pprint(dir(results))
pprint(results.model_dump())

pprint(results.web.model_dump().keys())

pprint(results.mixed.model_dump())

len(results.web_results)

pprint(results.model_dump())
pprint(results.model_dump()['web']['results'][-1])

