# StudentLife Location Analysis

This project analyzes GPS data from the StudentLife dataset to identify significant locations in students' lives. It uses clustering algorithms to identify places where students spend significant time and the Foursquare API to label these locations.

## Features

- Identifies significant locations using DBSCAN clustering
- Distinguishes between passing through and staying at locations
- Uses Foursquare API to get venue information
- Creates interactive map visualizations
- Generates daily pattern analysis plots
- Includes a test mode to avoid using API credits during development

## Requirements

- Python 3.8+

## Installation

1. Clone this repository
2. Create a `.env` file with your Foursquare API key:
   ```
   FOURSQUARE_API=your_api_key_here
   ```

## Usage

### Full Analysis
Run the main analysis script:
```bash
python analyze_locations.py
```

### Test Mode
To test the functionality without using API credits:
```bash
python test_locations.py
```

The test script uses a small sample of data (100 points) and mock venue information to demonstrate the functionality.

## Data Sources

- GPS data from the StudentLife dataset
- Venue information from Foursquare API

## Methodology

1. **Data Preprocessing**:
   - Load GPS data from CSV files
   - Convert timestamps to datetime objects
   - Clean and validate coordinates

2. **Significant Location Detection**:
   - Use DBSCAN clustering to identify clusters of GPS points
   - Calculate duration and stationary time for each cluster
   - Filter out noise points and temporary stops

3. **Venue Labeling**:
   - Use Foursquare API to get venue information for each significant location
   - Match venues based on coordinates and distance

4. **Visualization**:
   - Create interactive maps showing all GPS points and significant locations
   - Generate plots showing daily patterns of visits
   - Color-code points based on travel state (moving vs. stationary)

## Output

The script generates:
1. A summary of significant locations with:
   - Coordinates
   - Duration of visits
   - Stationary time
   - Venue information (if available)
2. Interactive map (`locations_map.html`) showing:
   - All GPS points (red for moving, blue for stationary)
   - Significant locations (green markers)
   - Popup information for each point
3. Daily pattern plots (`daily_patterns.png`) showing:
   - Hourly distribution of visits
   - Distribution by travel state
4. Location density heatmap (`location_heatmap.png`) showing:
   - Areas of high location density
   - Concentration of visits
5. Travel patterns plot (`travel_patterns.png`) showing:
   - Speed distribution
   - Travel state distribution

