# [Agriculture Data Analytics](./../../../)

## Maps

> "\[It's\] impossible to map the world – we select and make graphics so that we can understand it" - Roger Tomlinson, 1981

### GEO boundaries

[geoBoundaries](https://www.geoboundaries.org/): The world's largest open, free, and research-ready database of political administrative boundaries.

This folder contains files for a choropleth map of Ireland, downloaded from <https://www.geoboundaries.org/data/1_3_3/zip/shapefile/IRL/IRL_ADM1.shp.zip>

- Republic of Ireland: [irl-adm1](irl-adm1/)

### Example usage
```python
import geopandas
roi_geodataframe_filepath = "./../maps/irl-adm1/IRL_ADM1.shp"
roi_geodataframe = geopandas.read_file(roi_geodataframe_filepath)
```

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.