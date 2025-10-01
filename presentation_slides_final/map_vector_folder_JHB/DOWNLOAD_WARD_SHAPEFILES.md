# Where to Download Johannesburg Ward Boundary Shapefiles

## 1. **Municipal Demarcation Board (Primary Source)**
**URL:** http://www.demarcation.org.za/
- Go to "Data & Maps" → "Spatial Data Downloads"
- Look for "Ward Boundaries 2021" or latest year
- Select "Gauteng" → "City of Johannesburg"
- Download formats: Shapefile (.shp), KML, GeoJSON

## 2. **Statistics South Africa (Stats SA)**
**URL:** https://www.statssa.gov.za/?page_id=1021&id=city-of-johannesburg-municipality
- Navigate to "Geography" → "Spatial Data"
- Download "Ward Boundaries - City of Johannesburg"
- Includes census ward codes matching your GCRO data

## 3. **South African Municipal Data**
**URL:** https://municipalities.co.za/map/2/city-of-johannesburg-metropolitan-municipality
- Provides ward-level boundaries
- Interactive map with download options

## 4. **Open Data Portal - City of Johannesburg**
**URL:** https://data.joburg.org.za/
- Search for "ward boundaries" or "electoral wards"
- Usually available in multiple formats

## 5. **GCRO Data Portal** 
**URL:** https://data.gcro.ac.za/
- Quality of Life Survey spatial data
- Ward boundaries used in GCRO surveys
- Matches your GCRO dataset ward codes

## 6. **OpenStreetMap/Overpass Turbo**
**URL:** https://overpass-turbo.eu/
Query for Johannesburg wards:
```
[out:json];
area["name"="Johannesburg"]["admin_level"="8"];
(
  way["admin_level"="10"](area);
  relation["admin_level"="10"](area);
);
out geom;
```

## Recommended Download:
**For your ENBEL project, I recommend:**
1. **GCRO Data Portal** - Will match your ward codes exactly
2. **Municipal Demarcation Board** - Official boundaries

## File Names to Look For:
- `wards2021_jhb.shp`
- `COJ_Wards_2021.shp`
- `gauteng_wards.shp`
- `jhb_electoral_wards.shp`

## What You'll Get:
- `.shp` - Main shapefile
- `.shx` - Shape index
- `.dbf` - Attribute data (ward names, codes)
- `.prj` - Projection information
- `.cpg` - Character encoding

## After Downloading:
Place the files in your `map_vector_folder_JHB` directory and I can create a detailed map with actual ward boundaries showing:
- All 508 Johannesburg wards
- GCRO survey coverage by ward
- Clinical trial site locations overlaid
- Ward-level heat maps of data density