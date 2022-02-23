import serpapi as sp

search=sp.GoogleScholarSearch({"q":'coffee'})
data=search.get_html()

# source:The_Review_of_Financial_Studies