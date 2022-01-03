# [Agriculture Data Analytics](./../../../)

## Gantt Chart

>"Make a bad plan. Make the best one you can, but don't get obsessive about it. Make a plan, implement it. You'll figure out when you implement it why it's stupid, exactly, and then you can fix it a little bit, and then you can fix it a bit more, and then, eventually, you get a good plan, even if you start with something that's not the best." - [Dr. Jordan B. Peterson, 2018](https://www.jordanbpeterson.com/transcripts/aubrey-marcus/)  

[![](./images/gantt-chart.jfif)](https://tinyurl.com/y6mf7kvy)

> Gantt chart  

Links made with [Link Shortener Extension](https://timleland.com/link-shortener-extension/)  

Gantt chart made with [Mermaid](https://mermaid-js.github.io/mermaid-live-editor/edit/)  

### Gantt Chart Mermaid Code
```mermaid
gantt
 title Data Analytics Project Template - Gantt Chart
 dateFormat  YYYY-MM-DD
 axisFormat  %b-%e
section Setup
  Project Setup       :done,    setup,     2021-12-19,    2021-12-22
  Trello              :done,               2021-12-19,    2021-12-31 
  GitHub              :done,               2021-12-22,    2021-12-31
  IDE Setup           :done,               2021-12-22,    2022-01-03
 section Datasets
  Datasets            :active,  datasets,  2022-01-03,    2022-01-07
  Basic Stats         :active,             2022-01-03,    2022-01-07
  Sentiment Analysis  :active,             2022-01-03,    2022-01-07
 section Machine Learning
  Machine Learning    :active,  ml,        after datasets,    7d
 section Meetings
  Planning            :milestone,       2021-12-19,   1h
  Meeting             :milestone,       2021-12-22,   1h
  Sprint              :milestone,       2021-12-29,   1h
  Sprint              :milestone,       2022-01-03,   1h
  Staff Review        :crit,milestone,  2022-01-12,   1h
  Review              :milestone,       2022-01-26,   1h
  Submission          :crit,milestone,  2022-01-30,   1h
```

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.