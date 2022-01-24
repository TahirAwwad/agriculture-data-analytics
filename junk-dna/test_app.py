import streamlit as st
import streamlit.components.v1 as components
from owid import grapher, site

data = site.get_chart_data(slug="beef-and-buffalo-meat-production-tonnes")
st.dataframe(data.head())

chart = grapher.Chart(
    data
).encode(
    x="year",
    y="value",
    c="entity"
).select([
    "France",
    "Germany",
    "Netherlands"
]).label(
    title="Average height of men by year of birth"
).interact(
    entity_control=True,
    enable_map=True
)

components.html(chart._repr_html_(), height=400, scrolling=True)

HtmlFile = open("notebook-2-02-sa-bovine-tuberculosis.html",
                'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height=800, scrolling=True)
# components.iframe("path/to/page.html")
