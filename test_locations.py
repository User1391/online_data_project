import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv
import requests
import folium
from folium.plugins import HeatMap
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load environment variables
load_dotenv()

class LocationAnalyzer:
    def __init__(self):
        self.foursquare_api_key = os.getenv('FOURSQUARE_API')
        self.foursquare_base_url = "https://api.foursquare.com/v3"
        
    def load_gps_data(self, file_path, sample_size=100):
        """Load GPS data from CSV file and optionally sample it."""
        # Define the expected columns in the correct order
        columns = ['time', 'provider', 'network_type', 'accuracy', 'latitude', 'longitude', 'altitude', 'bearing', 'speed', 'travelstate']
        
        # Read CSV file with header, ensuring we use the correct columns
        df = pd.read_csv(file_path, usecols=columns)
        
        # force all empty values to be np.nan
        df = df.replace('', np.nan)
        
        # Convert numeric columns to float, ensuring we use the correct columns
        numeric_columns = ['accuracy', 'latitude', 'longitude', 'altitude', 'bearing', 'speed']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert time column to datetime (it's in Unix timestamp format)
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Fill NaN values in travelstate with 'unknown'
        df['travelstate'] = df['travelstate'].fillna('unknown')
        
        # Sample data if requested
        if sample_size:
            df = df.sort_values('time').head(sample_size)
        
        # Verify the coordinate ranges are correct (Dartmouth area)
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        print("\nCoordinate ranges:")
        print(f"Latitude range: {valid_coords['latitude'].min():.6f} to {valid_coords['latitude'].max():.6f}")
        print(f"Longitude range: {valid_coords['longitude'].min():.6f} to {valid_coords['longitude'].max():.6f}")
        
        return df
    
    def identify_significant_locations(self, df, eps=0.0001, min_samples=3):
        """Identify significant locations using DBSCAN clustering."""
        # Filter out rows with invalid coordinates
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        
        print("\nSample of input coordinates:")
        print(valid_coords[['latitude', 'longitude']].head())
        
        # Convert coordinates to radians for better clustering
        coords = valid_coords[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        
        print("\nSample of coordinates in radians:")
        print(coords_rad[:5])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(coords_rad)
        
        # Add cluster labels to dataframe
        valid_coords['cluster'] = clustering.labels_
        
        print("\nUnique cluster labels:", set(clustering.labels_))
        
        # Calculate cluster centers and durations
        significant_locations = []
        
        # First, handle noise points (cluster_id = -1)
        noise_points = valid_coords[valid_coords['cluster'] == -1]
        if not noise_points.empty:
            significant_locations.append({
                'cluster_id': -1,
                'latitude': noise_points['latitude'].mean(),
                'longitude': noise_points['longitude'].mean(),
                'duration': 0,  # Noise points are not considered for duration
                'stationary_time': 0,  # Noise points are not considered for stationary time
                'points': noise_points,
                'is_noise': True
            })
        
        # Then handle regular clusters
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise points as they're handled above
                continue
                
            cluster_points = valid_coords[valid_coords['cluster'] == cluster_id]
            
            # Calculate center (mean of lat/lon)
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            print(f"\nCluster {cluster_id} center:")
            print(f"Latitude: {center_lat}, Longitude: {center_lon}")
            print(f"Sample points in cluster:")
            print(cluster_points[['latitude', 'longitude']].head())
            
            # Calculate duration between first and last point
            time_range = cluster_points['time'].agg(['min', 'max'])
            duration = (time_range['max'] - time_range['min']).total_seconds() / 3600  # Convert to hours
            
            # Calculate stationary time (10 seconds per stationary point)
            stationary_points = cluster_points[cluster_points['travelstate'] == 'stationary']
            stationary_time = len(stationary_points) * 10 / 3600  # Convert to hours
            
            significant_locations.append({
                'cluster_id': cluster_id,
                'latitude': center_lat,
                'longitude': center_lon,
                'duration': duration,
                'stationary_time': stationary_time,
                'points': cluster_points,
                'is_noise': False
            })
        
        return significant_locations
    
    def get_venue_info(self, lat, lon, test_mode=False):
        """Get venue information from Foursquare API."""
        if test_mode:
            return {
                'name': 'Test Venue',
                'category': 'Test Category',
                'distance': 50,
                'address': 'Test Address'
            }
            
        headers = {
            "Accept": "application/json",
            "Authorization": self.foursquare_api_key
        }
        
        # Search for all nearby venues without category restriction
        params = {
            "ll": f"{lat},{lon}",
            "radius": 100,  # 100m radius for precise location matching
            "limit": 10,  # Get top 10 venues
            "sort": "DISTANCE",
            "fields": "name,categories,distance,location,fsq_id,geocodes,tel,website,hours,rating,stats,price,photos"
        }
        
        try:
            print(f"\nMaking API call for coordinates: {lat}, {lon}")
            print(f"Headers: {headers}")
            print(f"Params: {params}")
            
            response = requests.get(f"{self.foursquare_base_url}/places/nearby", headers=headers, params=params)
            
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text[:200]}...")
            
            # Handle rate limiting
            if response.status_code == 429:
                print("Rate limit exceeded. Waiting before retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
                response = requests.get(f"{self.foursquare_base_url}/places/nearby", headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # Get all venues
                    venues = data['results']
                    # Get the closest venue as primary
                    primary_venue = venues[0]
                    return {
                        'name': primary_venue.get('name', 'Unknown'),
                        'category': primary_venue.get('categories', [{}])[0].get('name', 'Unknown'),
                        'distance': primary_venue.get('distance', 0),
                        'address': primary_venue.get('location', {}).get('formatted_address', 'Unknown'),
                        'fsq_id': primary_venue.get('fsq_id', 'Unknown'),
                        'tel': primary_venue.get('tel', 'Unknown'),
                        'website': primary_venue.get('website', 'Unknown'),
                        'rating': primary_venue.get('rating', 'Unknown'),
                        'hours': primary_venue.get('hours', {}),
                        'stats': primary_venue.get('stats', {}),
                        'price': primary_venue.get('price', 'Unknown'),
                        'photos': primary_venue.get('photos', []),
                        'all_venues': venues  # Include all nearby venues
                    }
            else:
                print(f"Foursquare API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error fetching venue info: {e}")
            
        # Fallback to test data if API fails
        return {
            'name': 'Unknown Location',
            'category': 'Unknown Category',
            'distance': 0,
            'address': 'Address not available',
            'fsq_id': 'Unknown',
            'tel': 'Unknown',
            'website': 'Unknown',
            'rating': 'Unknown',
            'hours': {},
            'stats': {},
            'price': 'Unknown',
            'photos': [],
            'all_venues': []
        }

    def analyze_venue_patterns(self, significant_locations):
        """Analyze venue patterns to identify common locations."""
        venue_categories = {}
        venue_names = {}
        
        for loc in significant_locations:
            venue_info = self.get_venue_info(loc['latitude'], loc['longitude'], test_mode=False)
            if venue_info:
                # Count categories
                category = venue_info['category']
                venue_categories[category] = venue_categories.get(category, 0) + 1
                
                # Count venue names
                name = venue_info['name']
                venue_names[name] = venue_names.get(name, 0) + 1
        
        # Sort by frequency
        sorted_categories = sorted(venue_categories.items(), key=lambda x: x[1], reverse=True)
        sorted_names = sorted(venue_names.items(), key=lambda x: x[1], reverse=True)
        
        print("\nVenue Analysis:")
        print("\nMost common categories:")
        for category, count in sorted_categories[:5]:
            print(f"{category}: {count} locations")
            
        print("\nMost common venues:")
        for name, count in sorted_names[:5]:
            print(f"{name}: {count} visits")
        
        return sorted_categories, sorted_names
    
    def visualize_locations(self, df, significant_locations, output_file='locations_map.html', test_mode=True):
        """Create an interactive map visualization."""
        # Create base map centered on the mean location
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=15)  # Increased zoom level for better detail
        
        # Define colors for clusters
        cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']
        
        # Create feature groups for each cluster
        cluster_groups = {}
        
        # Create a feature group for noise points
        noise_group = folium.FeatureGroup(name='Noise Points')
        m.add_child(noise_group)
        
        # Add points to their respective cluster groups
        for loc in significant_locations:
            cluster_id = loc['cluster_id']
            color = cluster_colors[cluster_id % len(cluster_colors)]
            
            # Create feature group for cluster if it doesn't exist
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = folium.FeatureGroup(name=f'Location {cluster_id}')
                m.add_child(cluster_groups[cluster_id])
            
            # Get venue info
            venue_info = self.get_venue_info(loc['latitude'], loc['longitude'], test_mode)
            
            # Format time range
            time_range = loc['points']['time'].agg(['min', 'max'])
            time_range_str = f"{time_range['min'].strftime('%Y-%m-%d %H:%M')} to {time_range['max'].strftime('%Y-%m-%d %H:%M')}"
            
            # Create nearby venues HTML
            nearby_venues_html = "<br><b>Nearby Venues:</b><br>"
            for venue in venue_info.get('all_venues', []):
                categories = [cat.get('name', 'Unknown') for cat in venue.get('categories', [])]
                nearby_venues_html += f"""
                - {venue.get('name', 'Unknown')} ({', '.join(categories)})<br>
                  Distance: {venue.get('distance', 0)}m<br>
                  Address: {venue.get('location', {}).get('formatted_address', 'Unknown')}<br>
                """
            
            # Add cluster center marker with detailed popup
            popup_text = f"""
            <b>Location {cluster_id}</b><br>
            <b>Time Range:</b> {time_range_str}<br>
            <b>Duration:</b> {loc['duration']:.1f} hours<br>
            <b>Stationary time:</b> {loc['stationary_time']:.1f} hours<br>
            <b>Points at location:</b> {len(loc['points'])}<br>
            <b>Primary Venue:</b> {venue_info['name']}<br>
            <b>Category:</b> {venue_info['category']}<br>
            <b>Address:</b> {venue_info['address']}<br>
            <b>Phone:</b> {venue_info['tel']}<br>
            <b>Website:</b> {venue_info['website']}<br>
            <b>Rating:</b> {venue_info['rating']}<br>
            {nearby_venues_html}
            """
            
            # Add location center marker
            folium.CircleMarker(
                location=[loc['latitude'], loc['longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_opacity=1.0,
                popup=folium.Popup(popup_text, max_width=400),
                tooltip=f"Location {cluster_id} Center",
                z_index_offset=1000  # Ensure center markers are always on top
            ).add_to(cluster_groups[cluster_id])
            
            # Add all points in the cluster
            points = loc['points']
            for _, point in points.iterrows():
                folium.CircleMarker(
                    location=[point['latitude'], point['longitude']],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=f"Time: {point['time']}<br>State: {point['travelstate']}<br>Speed: {point['speed']:.1f} m/s",
                    tooltip=f"Location {cluster_id} Point",
                    z_index_offset=0  # Keep points below center markers
                ).add_to(cluster_groups[cluster_id])
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_file)
    
    def plot_daily_patterns(self, df, output_file='daily_patterns.png'):
        """Plot daily patterns of location visits."""
        # Extract hour from timestamp
        df['hour'] = df['time'].dt.hour
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Hourly distribution of visits
        sns.histplot(data=df, x='hour', bins=24, ax=ax1)
        ax1.set_title('Hourly Distribution of Location Visits')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Visits')
        
        # Plot 2: Hourly distribution by travel state
        # Filter out 'unknown' travel states and ensure we have valid data
        df_filtered = df[df['travelstate'].isin(['moving', 'stationary'])]
        if not df_filtered.empty:
            sns.histplot(data=df_filtered, x='hour', hue='travelstate', bins=24, ax=ax2)
            ax2.set_title('Hourly Distribution by Travel State')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Visits')
        else:
            ax2.text(0.5, 0.5, 'No valid travel state data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hourly Distribution by Travel State')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Number of Visits')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
    
    def plot_location_heatmap(self, df, output_file='location_heatmap.png'):
        """Create a heatmap of location density."""
        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=df, x='longitude', y='latitude', cmap='viridis', fill=True)
        plt.title('Location Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(output_file)
        plt.close()
    
    def plot_travel_patterns(self, df, output_file='travel_patterns.png'):
        """Plot travel patterns and speeds."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Speed distribution
        sns.histplot(data=df, x='speed', bins=50, ax=ax1)
        ax1.set_title('Speed Distribution')
        ax1.set_xlabel('Speed (m/s)')
        ax1.set_ylabel('Count')
        
        # Plot 2: Travel state distribution
        sns.countplot(data=df, x='travelstate', ax=ax2)
        ax2.set_title('Travel State Distribution')
        ax2.set_xlabel('Travel State')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def create_heatmap(self, df, output_file='locations_heatmap.html'):
        """Create a heatmap visualization of all GPS points."""
        # Create base map centered on the mean location
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], 
                      zoom_start=15)
        
        # Prepare data for heatmap
        # Filter out invalid coordinates and create list of [lat, lon, intensity]
        heat_data = []
        for _, row in df.dropna(subset=['latitude', 'longitude']).iterrows():
            # Use stationary points as higher intensity
            intensity = 2.0 if row['travelstate'] == 'stationary' else 1.0
            heat_data.append([row['latitude'], row['longitude'], intensity])
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            radius=15,  # Radius of each point
            blur=10,    # Blur factor
            max_zoom=13,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'red'}
        ).add_to(m)
        
        # Save map
        m.save(output_file)

def main():
    # Initialize analyzer
    analyzer = LocationAnalyzer()
    
    # Load GPS data for user 0 (sample of 100 points)
    df = analyzer.load_gps_data('dataset/sensing/gps/gps_u51.csv')
    
    # Identify significant locations with adjusted parameters
    # eps = 0.00000785 is approximately 50 meters at this latitude
    # min_samples = 3 means at least 3 points to form a cluster
    significant_locations = analyzer.identify_significant_locations(df, eps=0.00000785, min_samples=3)
    
    # Analyze venue patterns
    analyzer.analyze_venue_patterns(significant_locations)
    
    # Create visualizations
    analyzer.visualize_locations(df, significant_locations, test_mode=False)
    analyzer.create_heatmap(df)  # Add heatmap visualization
    analyzer.plot_daily_patterns(df)
    analyzer.plot_location_heatmap(df)
    analyzer.plot_travel_patterns(df)
    
    # Print summary
    print(f"\nFound {len(significant_locations)} significant locations:")
    for loc in significant_locations:
        venue_info = analyzer.get_venue_info(loc['latitude'], loc['longitude'], test_mode=False)
        print(f"\nLocation {loc['cluster_id']}:")
        print(f"Coordinates: ({loc['latitude']:.6f}, {loc['longitude']:.6f})")
        print(f"Duration: {loc['duration']:.1f} hours")
        print(f"Stationary time: {loc['stationary_time']:.1f} hours")
        if venue_info:
            print(f"Venue: {venue_info['name']}")
            print(f"Category: {venue_info['category']}")
            print(f"Address: {venue_info['address']}")
            print(f"Distance: {venue_info['distance']} meters")

if __name__ == "__main__":
    main() 