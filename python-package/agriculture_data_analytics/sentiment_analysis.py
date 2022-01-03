def preprocess(html_text: str):
    html_text = html_text.str.replace("(<br/>)", "")
    html_text = html_text.str.replace('(<a).*(>).*(</a>)', '')
    html_text = html_text.str.replace('(&amp)', '')
    html_text = html_text.str.replace('(&gt)', '')
    html_text = html_text.str.replace('(&lt)', '')
    html_text = html_text.str.replace('(\xa0)', ' ')  
    return html_text
