#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


df = pd.read_csv("./../artifacts/agriculture-ie-clean.csv")
text = df['clean_text']

#positive tokens

#postive word tokens sample

print('Random positive word tokens: \n')
cl = df.loc[df.polarity_tokens >0.5, ['clean_text']].sample(7).values
for c in cl:
    print(c[0])

df['polarity_tokens'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity_tokens',
    linecolor='black',
    yTitle='count',
    title='Token Polarity Distribution')

