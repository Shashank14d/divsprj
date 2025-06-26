"""
Dataset Integration Script for Smart AI-Based Diet and Workout Planner
This script helps integrate the user's Excel dataset into the application.
"""

import pandas as pd
import os
import sys
from data_processor import DataProcessor

def analyze_excel_file(file_path):
    """
    Analyze the Excel file to understand its structure
    
    Args:
        file_path (str): Path to the Excel file
    
    Returns:
        dict: Analysis results
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        analysis = {
            'file_path': file_path,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for expected columns based on user's dataset
        expected_columns = [
            'Name', 'Age', 'Weight(in Kg)', 'Height (in feet)', 
            'Goal in Fitness', 'Food Style', 'Health Issues', 
            'Contact Number', 'Workout Plans', 'Diet Plan'
        ]
        
        found_columns = []
        missing_columns = []
        
        for col in expected_columns:
            # Check for exact match or similar names
            exact_match = col in df.columns
            similar_matches = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
            
            if exact_match:
                found_columns.append(col)
            elif similar_matches:
                found_columns.append(f"{col} (similar: {similar_matches[0]})")
            else:
                missing_columns.append(col)
        
        analysis['found_columns'] = found_columns
        analysis['missing_columns'] = missing_columns
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}

def clean_and_transform_data(df):
    """
    Clean and transform the dataset to match application requirements
    
    Args:
        df (pandas.DataFrame): Original dataset
    
    Returns:
        pandas.DataFrame: Cleaned and transformed dataset
    """
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Rename columns to match application expectations
    column_mapping = {
        ' Weight(in Kg)': 'Weight_KG',
        'Height (in feet)': 'Height_Feet',
        'Goal in Fitness ': 'Fitness_Goal',
        'Food Style ': 'Food_Preference',
        'Any Health Issues ': 'Health_Issues',
        ' Contact Number ': 'Contact_Number',
        'Workout Plans': 'Workout_Plans',
        'Diet Plan': 'Diet_Plan',
        ' Age': 'Age'
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in cleaned_df.columns:
            cleaned_df[new_col] = cleaned_df[old_col]
    
    # Handle height data - convert to feet and inches if needed
    if 'Height_Feet' in cleaned_df.columns:
        # If height is in decimal feet, convert to feet and inches
        cleaned_df['Height_Inches'] = 0  # Default to 0 inches
        
        # Check if height values are decimal (e.g., 5.2 feet)
        for idx, height in enumerate(cleaned_df['Height_Feet']):
            if pd.notna(height):
                try:
                    height = float(height)
                    if height > 10:  # Likely in cm, convert to feet
                        height_feet = height / 30.48
                        cleaned_df.at[idx, 'Height_Feet'] = int(height_feet)
                        cleaned_df.at[idx, 'Height_Inches'] = int((height_feet % 1) * 12)
                    elif height % 1 != 0:  # Decimal feet, convert to feet and inches
                        feet = int(height)
                        inches = int((height % 1) * 12)
                        cleaned_df.at[idx, 'Height_Feet'] = feet
                        cleaned_df.at[idx, 'Height_Inches'] = inches
                    else:  # Whole number feet
                        cleaned_df.at[idx, 'Height_Feet'] = int(height)
                        cleaned_df.at[idx, 'Height_Inches'] = 0
                except (ValueError, TypeError):
                    # If conversion fails, set default values
                    cleaned_df.at[idx, 'Height_Feet'] = 5
                    cleaned_df.at[idx, 'Height_Inches'] = 0
    
    # Clean fitness goal values
    if 'Fitness_Goal' in cleaned_df.columns:
        # Preserve original values from the dataset
        cleaned_df['Fitness_Goal'] = cleaned_df['Fitness_Goal'].astype(str).str.strip()
        # Only clean obvious variations
        goal_mapping = {
            'weight loss': 'Weight Loss',
            'muscle gain': 'Muscle Gain',
            'stay fit': 'Stay Fit',
            'fitness': 'Stay Fit',
            'lose weight': 'Weight Loss',
            'gain muscle': 'Muscle Gain',
            'fat loss with muscle gain': 'Fat Loss with Muscle Gain',
            'just fat loss': 'Just Fat Loss',
            'weight gain': 'Weight Gain'
        }
        
        cleaned_df['Fitness_Goal'] = cleaned_df['Fitness_Goal'].apply(
            lambda x: goal_mapping.get(x.lower(), x) if pd.notna(x) and x != 'nan' else 'Weight Loss'
        )
    
    # Clean food preference values
    if 'Food_Preference' in cleaned_df.columns:
        # Preserve original values from the dataset
        cleaned_df['Food_Preference'] = cleaned_df['Food_Preference'].astype(str).str.strip()
        # Only clean obvious variations
        food_mapping = {
            'veg': 'Pure Veg',
            'vegetarian': 'Pure Veg',
            'non-veg': 'Non-Veg',
            'non vegetarian': 'Non-Veg',
            'vegan': 'Pure Veg'
        }
        
        cleaned_df['Food_Preference'] = cleaned_df['Food_Preference'].apply(
            lambda x: food_mapping.get(x.lower(), x) if pd.notna(x) and x != 'nan' else 'Non-Veg'
        )
    
    # Convert weight to numeric, handling any non-numeric values
    if 'Weight_KG' in cleaned_df.columns:
        cleaned_df['Weight_KG'] = pd.to_numeric(cleaned_df['Weight_KG'], errors='coerce')
        # Fill missing values with median
        median_weight = cleaned_df['Weight_KG'].median()
        cleaned_df['Weight_KG'] = cleaned_df['Weight_KG'].fillna(median_weight)
    
    # Convert age to numeric, handling any non-numeric values
    if 'Age' in cleaned_df.columns:
        cleaned_df['Age'] = pd.to_numeric(cleaned_df['Age'], errors='coerce')
        # Fill missing values with median
        median_age = cleaned_df['Age'].median()
        cleaned_df['Age'] = cleaned_df['Age'].fillna(median_age)
    
    # Add BMI calculations
    if 'Weight_KG' in cleaned_df.columns and 'Height_Feet' in cleaned_df.columns:
        cleaned_df['BMI'] = cleaned_df.apply(
            lambda row: calculate_bmi(row['Height_Feet'], row['Height_Inches'], row['Weight_KG'])
            if pd.notna(row['Weight_KG']) and pd.notna(row['Height_Feet'])
            else None, axis=1
        )
        
        # Add BMI category
        cleaned_df['BMI_Category'] = cleaned_df['BMI'].apply(
            lambda x: categorize_bmi(x) if pd.notna(x) else 'Normal'
        )
    
    # Add default body type (will be predicted by AI)
    cleaned_df['Body_Type'] = 'Mesomorph'  # Default, will be updated by AI
    
    # Add timestamp
    from datetime import datetime
    cleaned_df['Created_Date'] = datetime.now()
    
    return cleaned_df

def calculate_bmi(height_feet, height_inches, weight_kg):
    """Calculate BMI from height (feet, inches) and weight (kg)"""
    total_height_inches = (height_feet * 12) + height_inches
    height_meters = total_height_inches * 0.0254
    bmi = weight_kg / (height_meters ** 2)
    return round(bmi, 2)

def categorize_bmi(bmi):
    """Categorize BMI into health categories"""
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def integrate_dataset(file_path, output_dir='integrated_data'):
    """
    Integrate the Excel dataset into the application
    
    Args:
        file_path (str): Path to the Excel file
        output_dir (str): Output directory for processed data
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Analyze and clean the data
        print("Analyzing dataset...")
        analysis = analyze_excel_file(file_path)
        
        if 'error' in analysis:
            print(f"Error analyzing file: {analysis['error']}")
            return
        
        print(f"\nDataset Analysis:")
        print(f"Total records: {analysis['total_rows']}")
        print(f"Total columns: {analysis['total_columns']}")
        print(f"Columns found: {analysis['found_columns']}")
        print(f"Missing columns: {analysis['missing_columns']}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        for i, row in enumerate(analysis['sample_data']):
            print(f"Row {i+1}: {row}")
        
        # Process the data
        print(f"\nProcessing data...")
        
        # Clean and transform the data
        cleaned_df = clean_and_transform_data(df)
        
        # Initialize data processor with cleaned data
        processor = DataProcessor()
        processor.data = cleaned_df
        
        # Validate the data
        validation = processor.validate_data()
        print(f"\nData Validation:")
        print(f"Valid: {validation['valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        
        # Get statistics
        stats = processor.get_statistics()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            if key != 'columns':  # Skip columns list for cleaner output
                print(f"{key}: {value}")
        
        # Save processed data
        output_file = os.path.join(output_dir, 'processed_dataset.xlsx')
        processor.save_to_excel(output_file)
        print(f"\nProcessed data saved to: {output_file}")
        
        # Export for training
        training_dir = os.path.join(output_dir, 'training_data')
        processor.export_for_training(training_dir)
        print(f"Training data exported to: {training_dir}")
        
        # Create a summary report
        create_summary_report(analysis, validation, stats, output_dir)
        
        # Create a mapping file for reference
        create_column_mapping_file(output_dir)
        
        print(f"\n✅ Dataset integration completed successfully!")
        print(f"📁 Check the '{output_dir}' folder for processed files.")
        
        return processor.data
        
    except Exception as e:
        print(f"❌ Error integrating dataset: {e}")
        return None

def create_column_mapping_file(output_dir):
    """Create a file showing the column mapping"""
    mapping_file = os.path.join(output_dir, 'column_mapping.txt')
    
    with open(mapping_file, 'w') as f:
        f.write("COLUMN MAPPING REFERENCE\n")
        f.write("=" * 40 + "\n\n")
        f.write("Original Column -> Application Column\n")
        f.write("-" * 40 + "\n")
        f.write("Name -> Name\n")
        f.write("Age -> Age\n")
        f.write("Weight(in Kg) -> Weight_KG\n")
        f.write("Height (in feet) -> Height_Feet + Height_Inches\n")
        f.write("Goal in Fitness -> Fitness_Goal\n")
        f.write("Food Style -> Food_Preference\n")
        f.write("Health Issues -> Health_Issues\n")
        f.write("Contact Number -> Contact_Number\n")
        f.write("Workout Plans -> Workout_Plans\n")
        f.write("Diet Plan -> Diet_Plan\n")
        f.write("\nCalculated Columns:\n")
        f.write("BMI -> Calculated from height and weight\n")
        f.write("BMI_Category -> Underweight/Normal/Overweight/Obese\n")
        f.write("Body_Type -> Default: Mesomorph (will be predicted by AI)\n")
    
    print(f"📋 Column mapping saved to: {mapping_file}")

def create_summary_report(analysis, validation, stats, output_dir):
    """
    Create a summary report of the dataset integration
    
    Args:
        analysis (dict): Dataset analysis results
        validation (dict): Data validation results
        stats (dict): Statistical summary
        output_dir (str): Output directory
    """
    report_path = os.path.join(output_dir, 'integration_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DATASET INTEGRATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Source file: {analysis['file_path']}\n")
        f.write(f"Total records: {analysis['total_rows']}\n")
        f.write(f"Total columns: {analysis['total_columns']}\n\n")
        
        f.write("2. COLUMN ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Found columns:\n")
        for col in analysis['found_columns']:
            f.write(f"  ✓ {col}\n")
        
        f.write("\nMissing columns:\n")
        for col in analysis['missing_columns']:
            f.write(f"  ✗ {col}\n")
        
        f.write("\n3. DATA VALIDATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Valid: {validation['valid']}\n")
        if validation['errors']:
            f.write("Errors:\n")
            for error in validation['errors']:
                f.write(f"  - {error}\n")
        if validation['warnings']:
            f.write("Warnings:\n")
            for warning in validation['warnings']:
                f.write(f"  - {warning}\n")
        
        f.write("\n4. STATISTICAL SUMMARY\n")
        f.write("-" * 20 + "\n")
        for key, value in stats.items():
            if key != 'columns':
                f.write(f"{key}: {value}\n")
        
        f.write("\n5. NEXT STEPS\n")
        f.write("-" * 20 + "\n")
        f.write("1. Review the processed dataset\n")
        f.write("2. Train the CNN model with real data\n")
        f.write("3. Update the application to use the trained model\n")
        f.write("4. Test the application with real user data\n")
        f.write("5. Use the existing workout and diet plans from your dataset\n")
    
    print(f"📄 Summary report saved to: {report_path}")

def main():
    """
    Main function to run the dataset integration
    """
    print("🎯 Smart AI-Based Diet and Workout Planner")
    print("📊 Dataset Integration Tool")
    print("=" * 50)
    
    # Check if Excel file is provided as argument
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        # Look for Excel files in current directory
        excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
        
        if not excel_files:
            print("❌ No Excel files found in current directory.")
            print("Please place your Excel dataset file in this directory or provide the path as an argument.")
            return
        
        if len(excel_files) == 1:
            excel_file = excel_files[0]
            print(f"📁 Found Excel file: {excel_file}")
        else:
            print("📁 Multiple Excel files found:")
            for i, file in enumerate(excel_files, 1):
                print(f"  {i}. {file}")
            
            try:
                choice = int(input("\nSelect file number: ")) - 1
                excel_file = excel_files[choice]
            except (ValueError, IndexError):
                print("❌ Invalid selection. Please run the script again.")
                return
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"❌ File not found: {excel_file}")
        return
    
    print(f"\n🔄 Starting integration of: {excel_file}")
    print(f"📋 Expected columns: Name, Age, Weight(in Kg), Height (in feet), Goal in Fitness, Food Style, Health Issues, Contact Number, Workout Plans, Diet Plan")
    
    # Integrate the dataset
    integrated_data = integrate_dataset(excel_file)
    
    if integrated_data is not None:
        print(f"\n🎉 Integration completed!")
        print(f"📊 Your dataset is now ready to be used with the fitness planner application.")
        print(f"🌐 You can now run 'python app.py' to start the application with your real data.")
        print(f"💡 Your existing workout and diet plans will be preserved and can be used in the application.")

if __name__ == "__main__":
    main() 