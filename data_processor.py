"""
Data Processing Script for Smart AI-Based Diet and Workout Planner
Handles Excel data processing and preparation for the application.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataProcessor:
    def __init__(self, excel_file_path=None):
        """
        Initialize the data processor
        
        Args:
            excel_file_path (str): Path to the Excel file containing user data
        """
        self.excel_file_path = excel_file_path
        self.data = None
        
    def load_excel_data(self, file_path=None):
        """
        Load data from Excel file
        
        Args:
            file_path (str): Path to Excel file (optional, uses self.excel_file_path if not provided)
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if file_path:
            self.excel_file_path = file_path
            
        if not self.excel_file_path or not os.path.exists(self.excel_file_path):
            print("Excel file not found. Creating sample data structure...")
            return self.create_sample_data()
        
        try:
            # Try different Excel engines
            try:
                self.data = pd.read_excel(self.excel_file_path, engine='openpyxl')
            except:
                try:
                    self.data = pd.read_excel(self.excel_file_path, engine='xlrd')
                except:
                    self.data = pd.read_excel(self.excel_file_path, engine='odf')
            
            print(f"Successfully loaded data from {self.excel_file_path}")
            print(f"Data shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            print("Creating sample data structure...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample data structure for demonstration purposes
        
        Returns:
            pandas.DataFrame: Sample data
        """
        # Sample data structure based on the project requirements
        sample_data = {
            'Name': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'David Brown'],
            'Age': [25, 30, 28, 35, 22],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Height_Feet': [5, 5, 6, 5, 5],
            'Height_Inches': [10, 6, 2, 4, 11],
            'Weight_KG': [70, 55, 80, 65, 75],
            'BMI': [22.1, 19.8, 24.5, 23.1, 25.2],
            'BMI_Category': ['Normal', 'Normal', 'Normal', 'Normal', 'Overweight'],
            'Food_Preference': ['Non-Veg', 'Veg', 'Non-Veg', 'Vegan', 'Non-Veg'],
            'Fitness_Goal': ['Muscle Gain', 'Weight Loss', 'Stay Fit', 'Weight Loss', 'Muscle Gain'],
            'Body_Type': ['Mesomorph', 'Ectomorph', 'Mesomorph', 'Ectomorph', 'Endomorph'],
            'Health_Issues': ['None', 'None', 'None', 'Lactose Intolerant', 'None'],
            'Image_Path': ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg', 'sample5.jpg'],
            'Created_Date': [datetime.now()] * 5
        }
        
        self.data = pd.DataFrame(sample_data)
        print("Sample data created successfully")
        return self.data
    
    def calculate_bmi(self, height_feet, height_inches, weight_kg):
        """
        Calculate BMI from height and weight
        
        Args:
            height_feet (int): Height in feet
            height_inches (int): Height in inches
            weight_kg (float): Weight in kilograms
        
        Returns:
            float: Calculated BMI
        """
        total_height_inches = (height_feet * 12) + height_inches
        height_meters = total_height_inches * 0.0254
        bmi = weight_kg / (height_meters ** 2)
        return round(bmi, 2)
    
    def categorize_bmi(self, bmi):
        """
        Categorize BMI into health categories
        
        Args:
            bmi (float): BMI value
        
        Returns:
            str: BMI category
        """
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def add_bmi_columns(self):
        """
        Add BMI and BMI category columns to the dataset
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Calculate BMI for each row
        self.data['BMI'] = self.data.apply(
            lambda row: self.calculate_bmi(
                row['Height_Feet'], 
                row['Height_Inches'], 
                row['Weight_KG']
            ), axis=1
        )
        
        # Categorize BMI
        self.data['BMI_Category'] = self.data['BMI'].apply(self.categorize_bmi)
        
        print("BMI and BMI category columns added successfully")
    
    def validate_data(self):
        """
        Validate the data for completeness and correctness
        
        Returns:
            dict: Validation results
        """
        if self.data is None:
            return {'valid': False, 'errors': ['No data loaded']}
        
        errors = []
        warnings = []
        
        # Check for required columns
        required_columns = [
            'Name', 'Age', 'Gender', 'Height_Feet', 'Height_Inches', 
            'Weight_KG', 'Food_Preference', 'Fitness_Goal'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        for col in required_columns:
            if col in self.data.columns:
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    warnings.append(f"Column '{col}' has {missing_count} missing values")
        
        # Validate age range
        if 'Age' in self.data.columns:
            invalid_age = self.data[(self.data['Age'] < 13) | (self.data['Age'] > 100)]
            if len(invalid_age) > 0:
                warnings.append(f"Found {len(invalid_age)} records with age outside valid range (13-100)")
        
        # Validate height range
        if 'Height_Feet' in self.data.columns and 'Height_Inches' in self.data.columns:
            invalid_height = self.data[
                (self.data['Height_Feet'] < 3) | (self.data['Height_Feet'] > 8) |
                (self.data['Height_Inches'] < 0) | (self.data['Height_Inches'] > 11)
            ]
            if len(invalid_height) > 0:
                warnings.append(f"Found {len(invalid_height)} records with invalid height values")
        
        # Validate weight range
        if 'Weight_KG' in self.data.columns:
            invalid_weight = self.data[(self.data['Weight_KG'] < 30) | (self.data['Weight_KG'] > 300)]
            if len(invalid_weight) > 0:
                warnings.append(f"Found {len(invalid_weight)} records with weight outside valid range (30-300 kg)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_records': len(self.data)
        }
    
    def get_statistics(self):
        """
        Get statistical summary of the data
        
        Returns:
            dict: Statistical summary
        """
        if self.data is None:
            return {}
        
        stats = {
            'total_records': len(self.data),
            'columns': list(self.data.columns)
        }
        
        # BMI statistics
        if 'BMI' in self.data.columns:
            stats['bmi_stats'] = {
                'mean': round(self.data['BMI'].mean(), 2),
                'median': round(self.data['BMI'].median(), 2),
                'min': round(self.data['BMI'].min(), 2),
                'max': round(self.data['BMI'].max(), 2),
                'std': round(self.data['BMI'].std(), 2)
            }
        
        # BMI category distribution
        if 'BMI_Category' in self.data.columns:
            stats['bmi_distribution'] = self.data['BMI_Category'].value_counts().to_dict()
        
        # Gender distribution
        if 'Gender' in self.data.columns:
            stats['gender_distribution'] = self.data['Gender'].value_counts().to_dict()
        
        # Food preference distribution
        if 'Food_Preference' in self.data.columns:
            stats['food_preference_distribution'] = self.data['Food_Preference'].value_counts().to_dict()
        
        # Fitness goal distribution
        if 'Fitness_Goal' in self.data.columns:
            stats['fitness_goal_distribution'] = self.data['Fitness_Goal'].value_counts().to_dict()
        
        # Body type distribution
        if 'Body_Type' in self.data.columns:
            stats['body_type_distribution'] = self.data['Body_Type'].value_counts().to_dict()
        
        return stats
    
    def save_to_excel(self, output_path='processed_data.xlsx'):
        """
        Save processed data to Excel file
        
        Args:
            output_path (str): Output file path
        """
        if self.data is None:
            print("No data to save")
            return
        
        try:
            self.data.to_excel(output_path, index=False, engine='openpyxl')
            print(f"Data saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def filter_by_criteria(self, **kwargs):
        """
        Filter data by various criteria
        
        Args:
            **kwargs: Filter criteria (e.g., gender='Male', bmi_category='Normal')
        
        Returns:
            pandas.DataFrame: Filtered data
        """
        if self.data is None:
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        
        for column, value in kwargs.items():
            if column in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[column] == value]
        
        return filtered_data
    
    def export_for_training(self, output_dir='training_data'):
        """
        Export data in format suitable for model training
        
        Args:
            output_dir (str): Output directory for training data
        """
        if self.data is None:
            print("No data to export")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export metadata
        metadata = {
            'total_samples': len(self.data),
            'features': list(self.data.columns),
            'export_date': datetime.now().isoformat()
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export CSV for easy processing
        csv_path = os.path.join(output_dir, 'training_data.csv')
        self.data.to_csv(csv_path, index=False)
        
        print(f"Training data exported to {output_dir}")
        print(f"CSV file: {csv_path}")
        print(f"Metadata: {os.path.join(output_dir, 'metadata.json')}")

def main():
    """
    Main function to demonstrate data processing
    """
    # Initialize data processor
    processor = DataProcessor()
    
    # Load or create data
    data = processor.load_excel_data()
    
    # Add BMI calculations if not present
    if 'BMI' not in data.columns:
        processor.add_bmi_columns()
    
    # Validate data
    validation = processor.validate_data()
    print("\nData Validation Results:")
    print(f"Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Get statistics
    stats = processor.get_statistics()
    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save processed data
    processor.save_to_excel('processed_fitness_data.xlsx')
    
    # Export for training
    processor.export_for_training()

if __name__ == "__main__":
    main() 