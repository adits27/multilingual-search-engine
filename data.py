import wikipedia

def getWikiContent(search):
    wikipedia.set_lang("en")
    pageTitles = wikipedia.search(search, results = 3)
    summaries = []
    for pageTitle in pageTitles[:3]:
        if wikiPage := wikipedia.page(title=pageTitle, auto_suggest=False):
            if summary := wikiPage.summary:
                summaries.append(summary)
    
    if not summaries:
        return "Sorry! I couldn't find anything relevant on Wikipedia!"
    return "\n\n".join(summaries)