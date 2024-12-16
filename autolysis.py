# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai  # Ensure to install this library: pip install openai

# Utility function to provide basic data diagnostics
def data_diagnostics(dataframe):
    print("Running basic diagnostics...")
    numerical_summary = dataframe.describe()
    missing_data = dataframe.isnull().sum()

    # Extract numeric-only columns for correlation analysis
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_table = numeric_columns.corr() if not numeric_columns.empty else pd.DataFrame()

    print("Diagnostics completed successfully.")
    return numerical_summary, missing_data, correlation_table


# Detecting outliers using IQR-based approach
def outlier_inspection(dataframe):
    print("Identifying outliers using IQR...")
    numeric_data = dataframe.select_dtypes(include=[np.number])

    # Compute Q1, Q3, and IQR
    lower_quartile = numeric_data.quantile(0.25)
    upper_quartile = numeric_data.quantile(0.75)
    iqr_value = upper_quartile - lower_quartile

    # Identify the count of outliers
    outlier_count = ((numeric_data < (lower_quartile - 1.5 * iqr_value)) | 
                     (numeric_data > (upper_quartile + 1.5 * iqr_value))).sum()
    print("Outlier detection completed.")
    return outlier_count


def plot_visuals(correlation_table, outlier_info, dataframe, save_folder):
    print("Creating plots...")
    os.makedirs(save_folder, exist_ok=True)

    # Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_table, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Heatmap of Correlations')
    correlation_path = os.path.join(save_folder, 'heatmap.png')
    plt.savefig(correlation_path)
    plt.close()

    # Generate outlier count bar chart
    if not outlier_info.empty and outlier_info.sum() > 0:
        plt.figure(figsize=(10, 6))
        outlier_info.plot(kind='bar', color='orange')
        plt.title('Detected Outliers')
        plt.xlabel('Features')
        plt.ylabel('Outlier Count')
        outlier_chart_path = os.path.join(save_folder, 'outlier_chart.png')
        plt.savefig(outlier_chart_path)
        plt.close()
    else:
        print("No outliers found for plotting.")
        outlier_chart_path = None

    # Plot a distribution graph for the first numeric column
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(dataframe[numeric_columns[0]], kde=True, color='blue', bins=25)
        plt.title(f'Distribution of {numeric_columns[0]}')
        dist_plot_path = os.path.join(save_folder, 'distribution_plot.png')
        plt.savefig(dist_plot_path)
        plt.close()
    else:
        dist_plot_path = None

    print("Plot generation completed.")
    return correlation_path, outlier_chart_path, dist_plot_path


def generate_report(summary, missing, correlations, outliers, output_path):
    print("Compiling analysis report...")
    report_path = os.path.join(output_path, 'analysis_report.md')
    try:
        with open(report_path, 'w') as file:
            file.write("# Automated Data Analysis Report\n\n")
            file.write("## Summary Statistics\n")
            file.write(summary.to_markdown() + "\n\n")

            file.write("## Missing Data\n")
            for column, count in missing.items():
                file.write(f"- {column}: {count} missing values\n")
            file.write("\n")

            file.write("## Correlation Analysis\n")
            file.write("![Correlation Heatmap](heatmap.png)\n\n")

            if outliers.sum() > 0:
                file.write("## Outlier Detection\n")
                file.write("![Outlier Counts](outlier_chart.png)\n\n")

            file.write("## Additional Observations\n")
            file.write("This report provides statistical insights, correlation patterns, and potential anomalies in the dataset.")

        print("Report creation successful.")
        return report_path
    except Exception as e:
        print(f"Error during report writing: {e}")
        return None


# Interface with the LLM service to produce a narrative summary
def fetch_llm_summary(prompt_text, context_details):
    print("Requesting summary from language model...")
    try:
        token = os.getenv("AIPROXY_TOKEN")
        base_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        request_payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"{prompt_text}\nContext:\n{context_details}"}
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }

        response = requests.post(base_url, headers=headers, data=json.dumps(request_payload))
        if response.status_code == 200:
            output = response.json()['choices'][0]['message']['content'].strip()
            return output
        else:
            print("LLM API Error: Failed to generate summary.")
            return "No summary available."

    except Exception as err:
        print(f"Error communicating with LLM API: {err}")
        return "LLM generation failed."


def main(dataset_path):
    print("Initializing the data analysis pipeline...")

    try:
        data = pd.read_csv(dataset_path, encoding='ISO-8859-1')
        print("Dataset successfully loaded.")
    except Exception as load_error:
        print(f"Error loading dataset: {load_error}")
        return

    summary, missing, correlation = data_diagnostics(data)
    outliers = outlier_inspection(data)

    output_folder = "analysis_outputs"
    os.makedirs(output_folder, exist_ok=True)

    heatmap_path, outlier_chart_path, distribution_path = plot_visuals(correlation, outliers, data, output_folder)

    llm_prompt = "Create a concise narrative based on the data analysis results."
    llm_summary = fetch_llm_summary(llm_prompt, f"Summary: {summary}\nMissing Data: {missing}\nOutliers: {outliers}")

    report_path = generate_report(summary, missing, correlation, outliers, output_folder)
    if report_path:
        with open(report_path, 'a') as report:
            report.write("\n\n## Generated Summary\n")
            report.write(f"{llm_summary}\n")
        print(f"Full analysis pipeline completed. Report saved at {report_path}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python refactored_script.py <data_file>")
        sys.exit(1)
    main(sys.argv[1])
