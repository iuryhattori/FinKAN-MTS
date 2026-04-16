from src.pipeline.preprocessing.preprocess import process_data


root_folder = r"C:\Users\Pichau\OneDrive\Desktop\Project-1\KANMTS\data\01-Raw"
output_folder = r"C:\Users\Pichau\OneDrive\Desktop\Project-1\KANMTS\data\02-preprocessed"

if __name__ == '__main__':
    process_data(
        root_folder= root_folder,
        output_folder= output_folder
        
    )

