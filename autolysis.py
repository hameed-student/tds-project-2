# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "chardet", "requests", "seaborn", "matplotlib", "python-dotenv"]
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

class DataAnalyzer:
    def __init__(self, dataset_path, api_key):
        self.dataset_path = dataset_path
        self.api_key = api_key
        self.df = None
        self.headers_json = None
        self.profile = None
        self.output_dir = os.path.splitext(self.dataset_path)[0]
        self.ensure_output_dir()

    def ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def read_data(self):
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
        self.headers_json = json.dumps({"headers": self.df.columns.tolist()})

    def create_profile(self):
        self.profile = {
            "shape": self.df.shape,
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.apply(str).to_dict(),
            "numeric_summary": self.df.describe().to_dict(),
            "headers": self.headers_json,
            "sample_data": self.df.head(3).to_dict()
        }

    def generate_scatter_plot(self):
        try:
            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Given the following dataset analysis, suggest two numeric columns from the dataset that would make an interesting scatterplot. Return only the column names, separated by a comma. No explanation needed. If there is not enough numeric columns, return empty string only."
                        },
                        {
                            "role": "user",
                            "content": str(self.profile)
                        }
                    ]
                }
            )
            response_data = response.json()
            selected_columns = response_data['choices'][0]['message']["content"].split(',')
        except (KeyError, IndexError, TypeError):
            return

        selected_columns = [col.strip() for col in selected_columns if col.strip()]
        if len(selected_columns) != 2:
            return

        x_col, y_col = selected_columns

        if x_col not in self.df.columns or y_col not in self.df.columns:
            return
        if not pd.api.types.is_numeric_dtype(self.df[x_col]) or not pd.api.types.is_numeric_dtype(self.df[y_col]):
            return

        self.df[x_col] = pd.to_numeric(self.df[x_col], errors='coerce')
        self.df[y_col] = pd.to_numeric(self.df[y_col], errors='coerce')

        df_clean = self.df.dropna(subset=[x_col, y_col])

        if df_clean.empty:
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_clean, x=x_col, y=y_col)
        plt.title(f'Scatterplot between {x_col} and {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        x_safe = x_col.replace(" ", "")
        y_safe = y_col.replace(" ", "")
        plot_path = os.path.join(self.output_dir, f'{x_safe}_{y_safe}_scatterplot.png')
        plt.savefig(plot_path, dpi=50, bbox_inches='tight')
        plt.close()

    def generate_correlation_heatmap(self):
        num_df = self.df.select_dtypes(include=['number'])
        if num_df.shape[1] < 2:
            return
        corr_matrix = num_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')

        heatmap_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path, dpi=50, bbox_inches='tight')
        plt.close()

    def generate_cluster_plot(self):
        try:
            response = requests.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": '''Given the following dataset summary, which numeric columns are suitable for clustering?
    Exclude any IDs or non-informative columns. Please provide a maximum of 5 columns.
    Return only the column names, separated by a comma. No explanation needed
    also if the dataset is not suitable for clustering, return empty string only'''
                        },
                        {
                            "role": "user",
                            "content": str(self.profile)
                        }
                    ]
                }
            )
            response_data = response.json()
            columns = response_data['choices'][0]['message']['content'].split(',')
            selected_cols = [col.strip() for col in columns if col.strip()]
        except (requests.exceptions.RequestException, KeyError, IndexError, TypeError):
            return

        if len(selected_cols) < 2:
            return

        selected_cols = [col for col in selected_cols if col in self.df.columns]
        if len(selected_cols) < 2:
            return

        numeric_imputer = SimpleImputer(strategy='mean')
        self.df[selected_cols] = numeric_imputer.fit_transform(self.df[selected_cols])

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[selected_cols])

        if len(self.df) < 3:
            return

        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(scaled_data)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.df[selected_cols[0]], y=self.df[selected_cols[1]], hue=self.df['Cluster'], palette='viridis', s=100, alpha=0.7)
        plt.title('KMeans Clustering')
        plt.xlabel(selected_cols[0])
        plt.ylabel(selected_cols[1])
        plt.legend(title='Cluster')

        cluster_plot_path = os.path.join(self.output_dir, 'clustering_plot.png')
        plt.savefig(cluster_plot_path, dpi=72, bbox_inches='tight')
        plt.close()

    def encode_image(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def get_image_story(self, image_path):
        base64_img = self.encode_image(image_path)
        endpoint = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Create a detailed and engaging story based on this image. This is for a project and consider it as a report, don't specify things like date, language, prepared by etc. DO NOT attach any images, image links, or additional references to images in your response. Only include the narrative text based on the provided context, you can make appropriate headings and subheadings, but not too many. I am providing headers for context, but do not directly reference it in your narrative. Focus only on analyzing the data structure and overall trends.\n\n headers:{self.headers_json}."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            story = result['choices'][0]['message']['content']
            return story
        except (requests.exceptions.RequestException, KeyError, IndexError):
            print(f"Failed to generate story for image {image_path}")
            return None

    def generate_readme(self):
        if not os.path.exists(self.output_dir):
            print(f"Directory does not exist: {self.output_dir}")
            return

        image_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        if not image_files:
            print("No PNG images found in the output directory.")
            return

        readme_content = "# Image Narratives\n\n"

        for image in image_files:
            image_path = os.path.join(self.output_dir, image)
            print(f"Processing image: {image_path}")

            story = self.get_image_story(image_path)
            if story:
                readme_content += f"## {os.path.splitext(image)[0]}\n\n"
                readme_content += f"![{image}](./{image})\n\n"
                readme_content += f"{story}\n\n"
            else:
                print(f"Could not generate story for image: {image}")

        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as readme_file:
            readme_file.write(readme_content)
        print(f"README.md created at {readme_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: started")
        sys.exit(1)

    try:
        load_dotenv()
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        raise ValueError("AIPROXY_TOKEN environment variable not set.")

    # Process each dataset file provided as an argument
    for dataset_file in sys.argv[1:]:
        print(f"Processing {dataset_file}")
        
        analyzer = DataAnalyzer(dataset_file, api_key)
        analyzer.read_data()
        analyzer.extract_headers()
        analyzer.create_profile()
        analyzer.generate_scatter_plot()
        analyzer.generate_correlation_heatmap()
        analyzer.generate_cluster_plot()
        analyzer.generate_readme()

        print(f"Finished processing {dataset_file}\n")
