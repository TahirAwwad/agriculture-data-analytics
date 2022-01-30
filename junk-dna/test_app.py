from owid import grapher, site
import streamlit
import streamlit.components.v1 as components

data = site.get_chart_data(slug="beef-and-buffalo-meat-production-tonnes")
streamlit.dataframe(data.head())

chart = grapher.Chart(data
).encode(x="year", y="value", c="entity"
).select(["France", "Germany", "Netherlands"]
).label(title="Beef and Buffalo meat production tonnes"
).interact(entity_control=True, enable_map=True
)

components.html(chart._repr_html_(), height=400, scrolling=True)

# components.iframe("path/to/page.html")
