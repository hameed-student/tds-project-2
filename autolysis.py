# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv","tempfile"]
# ///
import os
import sys
import json
import base64
import chardet
import requests
import subprocess
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from PIL import Image
import tempfile

class DataAnalyzer:
    """
    A class for comprehensive data analysis, visualization, and narrative generation
    using external APIs and standard Python libraries.
    """

    def __init__(self, dataset_path, api_key):
        """Initialize the DataAnalyzer with the dataset path and API key."""
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.df = None
        self.headers_json = None
        self.profile = None
        self.output_dir = os.getcwd()
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def read_data(self):
        """Load the dataset and detect its encoding."""
        try:
            with open(self.dataset_path, 'rb') as file:
                result = chardet.detect(file.read())
                encoding = result['encoding']

            self.df = pd.read_csv(self.dataset_path, encoding=encoding)
            if self.df is None or self.df.empty:
                sys.exit("Dataset is empty or could not be loaded.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    def extract_headers(self):
        """Extract and store the dataset headers as JSON."""
        self.headers_json = json.dumps({"headers": self.df.columns.tolist()})

    def create_profile(self):
        """Create a detailed profile of the dataset."""
        self.profile = {
            "shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.apply(str).to_dict(),
            "numeric_summary": self.df.describe().to_dict(),
            "headers": self.headers_json,
            "sample_data": self.df.head(3).to_dict()
        }

    def generate_visualization(self, kind):
        """Generate visualizations (scatter plot or heatmap) based on dataset analysis."""
        try:
            if kind == "scatter":
                self.generate_scatter_plot()
            elif kind == "heatmap":
                self.generate_correlation_heatmap()
            elif kind == "cluster":
                self.generate_cluster_plot()
        except Exception as e:
            print(f"Error generating {kind}: {e}")

    def generate_scatter_plot(self):
        """Generate a scatter plot based on API-selected columns."""
        # Improved error handling and efficiency for column selection
        try:
            selected_columns = self.get_suggested_columns("scatter")
            if len(selected_columns) != 2:
                return

            x_col, y_col = selected_columns
            if not pd.api.types.is_numeric_dtype(self.df[x_col]) or not pd.api.types.is_numeric_dtype(self.df[y_col]):
                return

            self.df[[x_col, y_col]] = self.df[[x_col, y_col]].apply(pd.to_numeric, errors='coerce')
            df_clean = self.df.dropna(subset=[x_col, y_col])
            if df_clean.empty:
                return

            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_clean, x=x_col, y=y_col)
            plt.title(f'Scatterplot between {x_col} and {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)

            
            plt.savefig('scatterplot.png', dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Scatter plot generation failed: {e}")
    
    def generate_cluster_plot(self):
        """Generate a cluster plot based on API-selected columns."""
        try:
            # Select columns for clustering
            selected_columns = self.get_suggested_columns("cluster")
            if len(selected_columns) < 2:
                return

            # Ensure selected columns are numeric
            numeric_cols = [col for col in selected_columns if pd.api.types.is_numeric_dtype(self.df[col])]
            if len(numeric_cols) < 2:
                return

            # Prepare the data
            df_clean = self.df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if df_clean.empty:
                return

            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)

            # Perform K-means clustering (use 3 clusters by default)
            kmeans = KMeans(n_clusters=3, random_state=42)
            df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

            # Create the cluster plot
            plt.figure(figsize=(10, 8))
            
            # If more than 2 columns, use first two for visualization
            scatter = sns.scatterplot(
                data=df_clean, 
                x=numeric_cols[0], 
                y=numeric_cols[1], 
                hue='Cluster', 
                palette='viridis'
            )

            plt.title(f'Cluster Plot of {numeric_cols[0]} vs {numeric_cols[1]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])

            plt.savefig('cluster_plot.png', dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Cluster plot generation failed: {e}")

    def generate_correlation_heatmap(self):
        """Generate a correlation heatmap."""
        try:
            num_df = self.df.select_dtypes(include=['number'])
            if num_df.shape[1] < 2:
                return

            corr_matrix = num_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Correlation Heatmap')

            
            plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Correlation heatmap generation failed: {e}")

    def get_suggested_columns(self, analysis_type):
        """Fetch suggested columns for analysis via API."""
        prompt_map = {
            "scatter": "Which 2 numeric columns are suitable for scatterplot? provide exactly 2 Your response should be a comma-separated list of column names only. the column names should be exactly as they appear in the dataset. return empty string if no columns are suitable for scatter plot..",
            "cluster": "Which numeric columns are suitable for clustering? Provide up to 5. Your response should be a comma-separated list of column names only. the column names should be exactly as they appear in the dataset. return empty string if no columns are suitable for clustering."}

        try:
            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": prompt_map[analysis_type]},
                        {"role": "user", "content": str(self.profile)}
                    ]
                }
            )
            response_data = response.json()
            columns = response_data['choices'][0]['message']['content'].split(',')
            print(columns)
            return [col.strip() for col in columns if col.strip()]
        except Exception as e:
            print(f"Error fetching {analysis_type} columns: {e}")
            return []

    def generate_readme(self):
        """Generate a README.md file containing narratives for visualizations."""
        try:
            image_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
            if not image_files:
                return

            readme_content = "# Data Analysis Visualizations\n\n"

            for image in image_files:
                image_path = os.path.join(self.output_dir, image)
                story = self.get_image_story(image_path)
                if story:
                    readme_content += f"## {os.path.splitext(image)[0]}\n\n" \
                                      f"![{image}](./{image})\n\n{story}\n\n"

            with open(os.path.join(self.output_dir, "README.md"), "w", encoding="utf-8") as readme_file:
                readme_file.write(readme_content)
        except Exception as e:
            print(f"README generation failed: {e}")

    def get_image_story(self, image_path):
        """Generate a narrative based on the image."""
        try:
            base64_img = self.compress_image(image_path)
            

            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Create a narrative based on this visualization."},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Describe this visualization"},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                            ]
                        }
                    ]
                }
            )
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Failed to generate story: {e}")
            return ""

    def compress_image(self,image_path, max_size=(800, 800), quality=85):
        """
    Compress an image while maintaining aspect ratio
"""
        with Image.open(image_path) as img:
            # Resize image if it's larger than max_size
            img.thumbnail(max_size, Image.LANCZOS)
            
            # Use a temporary file to handle the buffer
            with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as temp_file:
                img.save(temp_file.name, format="PNG", optimize=True, quality=quality)
                
                # Read the file contents and encode
                with open(temp_file.name, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset_path>")
        sys.exit(1)

    load_dotenv()
    try:
        api_key = os.environ["AI_PROXY"]
    except KeyError:
        print("AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

    for dataset_file in sys.argv[1:]:
        print(f"Processing {dataset_file}...")
        
        analyzer = DataAnalyzer(dataset_file, api_key)
        analyzer.read_data()
        analyzer.extract_headers()
        analyzer.create_profile()
        analyzer.generate_visualization("scatter")
        analyzer.generate_visualization("cluster")
        analyzer.generate_visualization("heatmap")
        analyzer.generate_readme()


