import pandas as pd
import matplotlib.pyplot as plt
import folium

df = pd.read_csv('data/train.csv')

print(len(df['scientific_name']))

#Number of unique birds by scientific name
print(df['scientific_name'].nunique())

#Top 10 most common birds
print(df['scientific_name'].value_counts().head(10))

# Drop rows without coordinates
df = df.dropna(subset=['latitude', 'longitude'])

# Create a base map centered around the mean location
map_center = [df['latitude'].mean(), df['longitude'].mean()]
bird_map = folium.Map(location=map_center, zoom_start=2)

# Add points to the map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        popup=row.get('common_name', 'Unknown'),
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(bird_map)

# Save map to HTML file
bird_map.save("bird_sightings_map.html")