# [Agriculture Data Analytics](./../../../)

## Gantt Chart

>"Make a bad plan. Make the best one you can, but don't get obsessive about it. Make a plan, implement it. You'll figure out when you implement it why it's stupid, exactly, and then you can fix it a little bit, and then you can fix it a bit more, and then, eventually, you get a good plan, even if you start with something that's not the best." - [Dr. Jordan B. Peterson, 2018](https://www.jordanbpeterson.com/transcripts/aubrey-marcus/)  

[![](./images/gantt-chart.jfif)](https://tinyurl.com/y6yvnn6c)

> Gantt chart  

Links made with [Link Shortener Extension](https://timleland.com/link-shortener-extension/)  

Gantt chart made with [Mermaid](https://mermaid-js.github.io/mermaid-live-editor/edit/)  


### Gantt Chart Mermaid Code
```mermaid
gantt
 title Data Analytics Project Template - Gantt Chart
 dateFormat  YYYY-MM-DD
 axisFormat  %b-%e
 section Guides
  Project Structure Guide    :done,    doc01,  2021-12-23,   1d
  Code Style                 :done,            2021-12-23,   1d
  Jupyter notebook layout    :done,            2021-12-23,   1d
  Build Python Package       :done,            2021-12-23,   1d
  Build Requirements.txt     :done,            2021-12-23,   1d
 section Dev Environment
  Install nbautoexport       :done,    doc06,  after doc01,  1d
 section Customize
  Assessment criteria        :active,  doc07,  after doc01,  1d
  Knowledge skills abilities :active,          after doc01,  1d
  Install Python Package     :active,          after doc07,  2d
  "Gantt chart"              :active,          after doc07,  2d
  Notebook managers          :active,  doc11,  after doc07,  2d
 section Notebooks
  bad example                :done,            after doc11,  2d
  bad example notes          :done,    doc13,  after doc11,  2d
  good example               :done,            after doc13,  1d
 section Meetings
  Planning                   :milestone,       2021-12-23,   1h
  Review                     :milestone,       2021-12-26,   1h
  Release                    :crit,milestone,  2021-12-30,   1h
```

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.