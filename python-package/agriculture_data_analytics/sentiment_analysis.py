def preprocess(html_text: str) ->str :
    """
    This function takes in a string of html text and returns a string of text
    with all html tags removed.
    html_text: string of html text
    return: string of text with all html tags removed
    """
    html_text = html_text.str.replace("(<br/>)","")
    html_text = html_text.str.replace('(<a).(>).(</a>)', '')
    html_text = html_text.str.replace('(&amp)', '')
    html_text = html_text.str.replace('(&gt)', '')
    html_text = html_text.str.replace('(&lt)', '')
    html_text = html_text.str.replace('(\xa0)', ' ')  
    return html_text
