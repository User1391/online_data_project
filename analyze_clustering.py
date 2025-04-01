import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from test_locations import LocationAnalyzer

class ClusteringAnalyzer:
    def __init__(self):
        self.analyzer = LocationAnalyzer()
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare GPS data for analysis."""
        df = self.analyzer.load_gps_data(file_path, sample_size=None)  # Load all points
        valid_coords = df.dropna(subset=['latitude', 'longitude'])
        coords = valid_coords[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        return df, valid_coords, coords, coords_rad
    
    def calculate_cluster_metrics(self, coords_rad, labels):
        """Calculate various clustering quality metrics."""
        # Remove noise points for metric calculation
        mask = labels != -1
        if sum(mask) > 1:  # Need at least 2 points for metrics
            metrics = {
                'silhouette': silhouette_score(coords_rad[mask], labels[mask]),
                'calinski_harabasz': calinski_harabasz_score(coords_rad[mask], labels[mask]),
                'davies_bouldin': davies_bouldin_score(coords_rad[mask], labels[mask])
            }
        else:
            metrics = {
                'silhouette': 0,
                'calinski_harabasz': 0,
                'davies_bouldin': 0
            }
        return metrics
    
    def analyze_noise_points(self, df, labels):
        """Analyze characteristics of noise points."""
        noise_mask = labels == -1
        noise_points = df[noise_mask]
        
        noise_analysis = {
            'total_points': len(df),
            'noise_points': len(noise_points),
            'noise_percentage': (len(noise_points) / len(df)) * 100,
            'noise_speed_mean': noise_points['speed'].mean(),
            'noise_speed_std': noise_points['speed'].std(),
            'noise_travel_states': noise_points['travelstate'].value_counts().to_dict()
        }
        
        return noise_analysis
    
    def analyze_cluster_characteristics(self, df, labels):
        """Analyze characteristics of each cluster."""
        cluster_stats = []
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = labels == cluster_id
            cluster_points = df[cluster_mask]
            
            # Calculate time-based metrics
            time_range = cluster_points['time'].agg(['min', 'max'])
            duration = (time_range['max'] - time_range['min']).total_seconds() / 3600  # hours
            
            # Calculate spatial metrics
            center_lat = cluster_points['latitude'].mean()
            center_lon = cluster_points['longitude'].mean()
            
            # Calculate distances from center
            distances = np.sqrt(
                (cluster_points['latitude'] - center_lat)**2 +
                (cluster_points['longitude'] - center_lon)**2
            )
            
            stats = {
                'cluster_id': cluster_id,
                'num_points': len(cluster_points),
                'duration_hours': duration,
                'mean_distance_from_center': distances.mean(),
                'max_distance_from_center': distances.max(),
                'stationary_percentage': (cluster_points['travelstate'] == 'stationary').mean() * 100,
                'mean_speed': cluster_points['speed'].mean(),
                'std_speed': cluster_points['speed'].std()
            }
            
            cluster_stats.append(stats)
            
        return pd.DataFrame(cluster_stats)
    
    def analyze_temporal_consistency(self, df, labels):
        """Analyze temporal consistency of clusters."""
        temporal_stats = []
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = labels == cluster_id
            cluster_points = df[cluster_mask].sort_values('time')
            
            # Calculate time gaps between consecutive points
            time_gaps = cluster_points['time'].diff().dt.total_seconds() / 3600  # hours
            
            stats = {
                'cluster_id': cluster_id,
                'mean_time_gap': time_gaps.mean(),
                'max_time_gap': time_gaps.max(),
                'std_time_gap': time_gaps.std(),
                'total_duration': (cluster_points['time'].max() - cluster_points['time'].min()).total_seconds() / 3600
            }
            
            temporal_stats.append(stats)
            
        return pd.DataFrame(temporal_stats)
    
    def plot_analysis_results(self, cluster_stats, temporal_stats, noise_analysis, output_dir='analysis_plots'):
        """Create visualization plots for the analysis results."""
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Cluster sizes
        plt.figure(figsize=(10, 6))
        sns.barplot(data=cluster_stats, x='cluster_id', y='num_points')
        plt.title('Number of Points per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Points')
        plt.savefig(f'{output_dir}/cluster_sizes.png')
        plt.close()
        
        # Plot 2: Distance from center
        plt.figure(figsize=(10, 6))
        # Melt the data to create a long format suitable for boxplot
        distance_data = cluster_stats[['mean_distance_from_center', 'max_distance_from_center']].melt()
        sns.boxplot(data=distance_data, x='variable', y='value')
        plt.title('Distance from Cluster Center')
        plt.xlabel('Distance Type')
        plt.ylabel('Distance (degrees)')
        plt.xticks([0, 1], ['Mean Distance', 'Max Distance'])
        plt.savefig(f'{output_dir}/distance_from_center.png')
        plt.close()
        
        # Plot 3: Temporal gaps
        plt.figure(figsize=(10, 6))
        # Melt the data to create a long format suitable for boxplot
        time_data = temporal_stats[['mean_time_gap', 'max_time_gap']].melt()
        sns.boxplot(data=time_data, x='variable', y='value')
        plt.title('Time Gaps Between Points')
        plt.xlabel('Time Gap Type')
        plt.ylabel('Time Gap (hours)')
        plt.xticks([0, 1], ['Mean Time Gap', 'Max Time Gap'])
        plt.savefig(f'{output_dir}/time_gaps.png')
        plt.close()
        
        # Plot 4: Travel state distribution
        plt.figure(figsize=(10, 6))
        travel_states = pd.Series(noise_analysis['noise_travel_states'])
        travel_states.plot(kind='bar')
        plt.title('Travel State Distribution in Noise Points')
        plt.xlabel('Travel State')
        plt.ylabel('Count')
        plt.savefig(f'{output_dir}/noise_travel_states.png')
        plt.close()
    
    def analyze_clustering(self, file_path, eps=0.00000785, min_samples=3):
        """Perform comprehensive clustering analysis."""
        # Load and prepare data
        df, valid_coords, coords, coords_rad = self.load_and_prepare_data(file_path)
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(coords_rad)
        labels = clustering.labels_
        
        # Calculate metrics
        metrics = self.calculate_cluster_metrics(coords_rad, labels)
        noise_analysis = self.analyze_noise_points(valid_coords, labels)
        cluster_stats = self.analyze_cluster_characteristics(valid_coords, labels)
        temporal_stats = self.analyze_temporal_consistency(valid_coords, labels)
        
        # Create visualizations
        self.plot_analysis_results(cluster_stats, temporal_stats, noise_analysis)
        
        # Print summary
        print("\nClustering Analysis Summary:")
        print(f"\nTotal points: {len(df)}")
        print(f"Valid points: {len(valid_coords)}")
        print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print(f"Number of noise points: {noise_analysis['noise_points']}")
        print(f"Noise percentage: {noise_analysis['noise_percentage']:.2f}%")
        
        print("\nCluster Quality Metrics:")
        print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.4f}")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.4f}")
        
        print("\nNoise Point Analysis:")
        print(f"Mean speed: {noise_analysis['noise_speed_mean']:.2f} m/s")
        print(f"Speed std: {noise_analysis['noise_speed_std']:.2f} m/s")
        print("\nTravel state distribution in noise points:")
        for state, count in noise_analysis['noise_travel_states'].items():
            print(f"{state}: {count}")
        
        print("\nCluster Statistics:")
        print(cluster_stats.describe())
        
        print("\nTemporal Statistics:")
        print(temporal_stats.describe())
        
        return {
            'metrics': metrics,
            'noise_analysis': noise_analysis,
            'cluster_stats': cluster_stats,
            'temporal_stats': temporal_stats
        }

def main():
    analyzer = ClusteringAnalyzer()
    results = analyzer.analyze_clustering('dataset/sensing/gps/gps_u00.csv')
    
    # Save results to CSV files
    results['cluster_stats'].to_csv('cluster_statistics.csv', index=False)
    results['temporal_stats'].to_csv('temporal_statistics.csv', index=False)

if __name__ == "__main__":
    main() 